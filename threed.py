from collections.abc import Iterable
import numpy as np
import pythreejs as p3j
import PIL.Image                     # for importing PNG files

#======================================================================
#
#  CRUCIAL HELPER FUNCTIONS
#
#=====================================================================  
def isArrayLike (a): return isinstance(a,Iterable) and not isinstance(a,str)
def normalize (v): return v/np.linalg.norm(v)
def copysign (x, s): return x * np.sign(x*s)
    
def findNormal (u):
    '''
    Return any vector that is perpendicular to the vector u
    '''
    if u[1]!=0: return normalize(np.cross ([1,0,0], u))
    if u[2]!=0: return normalize(np.cross ([0,1,0], u))
    if u[0]!=0: return normalize(np.cross ([0,0,1], u)) 
    raise Exception ('findNormal([0,0,0]) is illegal!')

def lookAt (forward,up):
    '''
    Return a 3x3 rotation matrix such that [0,0,1] maps to *forward* 
    and [0,1,0] maps to a direction close to *up*
    '''
    c = -normalize(forward)
    a = normalize(np.cross(up, c)) # not sure about all this
    b = normalize(np.cross(c, a))
    return np.array ([a,b,c]).T

def quatUToV (u,v):
    '''
    Return a quaternion that rotates vector u to vector v
    '''
    u = normalize(u); v = normalize(v); udotv = u@v
    if udotv==1:    return (0,0,0,1)                                   # identity
    elif udotv==-1: t = np.pi; a = findNormal (u)                      # rotate by PI about any axis perpendicular to u
    else:           t = np.arccos(udotv); a = normalize(np.cross(u,v)) # rotate by arccos(u.v) about uxv
    s = np.sin(t/2); c = np.cos(t/2)
    return (s*a[0], s*a[1], s*a[2], c)                                 # convert axis-angle (a,t) to quaternion

def quatFromRotMat (m):
    '''
    Return a quaternion corresponding to the 3x3 rotation matrix m
    '''
    w = np.sqrt( max( 0, 1 + m[0,0] + m[1,1] + m[2,2] ) ) / 2
    x = np.sqrt( max( 0, 1 + m[0,0] - m[1,1] - m[2,2] ) ) / 2
    y = np.sqrt( max( 0, 1 - m[0,0] + m[1,1] - m[2,2] ) ) / 2
    z = np.sqrt( max( 0, 1 - m[0,0] - m[1,1] + m[2,2] ) ) / 2
    x = copysign( x, m[2,1] - m[1,2] )
    y = copysign( y, m[0,2] - m[2,0] )
    z = copysign( z, m[1,0] - m[0,1] )
    return (x,y,z,w)

def chooseCameraOrientation (elev=0., azim=0., up=[0., 0., 1.]):
    '''
    Return a 3x3 rotation matrix corresponding to the camera orientation specified by:
        elev = elevation in radians
        azim = azimuth in radians
        up = up direction.
    The columns of the rotation matrix correspond to the right, up, and back directions according to the camera.
    Do not set elev to +/- pi/2.
    '''
    return lookAt (-np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)]), up)

def chooseCameraTargetAndDist (aspect, fov, ori, points, zoomOut=1.0):
    '''
    Calculate the position (3D vector) corresponding to these camera settings:
        ori       camera orientation (3x3 matrix)
        fov       camera's vertical field of view in radians
        aspect    camera aspect ratio
        points    array of points to be captured within camera view
    '''
    unl = points @ ori.T                  # object points in camera space (for some reason this involves ori.T)
    umin,vmin,wmin = np.min (unl, axis=0)  # bounding box of... 
    umax,vmax,wmax = np.max (unl, axis=0)  # ...object point coordinates in camera space
    camTgt = [(umax+umin)/2, (vmax+vmin)/2, (wmax+wmin)/2] @ ori   # transform back using inverse mtrix
    urange,vrange,wrange = np.abs([umax-umin,vmax-vmin,wmax-wmin])
    b = 2*np.tan(fov/2); a = b*aspect
    w0 = wrange/2 + max(urange/a, vrange/b)   # step back from plane of frontmost point
    w0 *= zoomOut                             # scale by additional safety factor
    if w0==0: w0 += 10                        # prevent it from being 0
    camDist = w0
    return camTgt,camDist
    #camPos = camTgt + w0 * ori[:,2]         
    
def chooseCameraDist (aspect, fov, ori, points, target, zoomOut=1.0):
    '''
    Calculate the position (3D vector) corresponding to these camera settings:
        ori       camera orientation (3x3 matrix)
        fov       camera's vertical field of view in radians
        aspect    camera aspect ratio
        points    array of points to be captured within camera view
        target    specified camera target (to be at center of view)
    '''
    unl = (points - np.array(target)) @ ori.T
    urange,vrange,wrange = np.max (np.abs(unl), axis=0)
    b = np.tan(fov/2); a = b*aspect
    w0 = wrange/2 + max(urange/a, vrange/b)   # step back from plane of frontmost point
    w0 *= zoomOut                             # scale by additional safety factor
    if w0==0: w0 += 10                        # prevent it from being 0
    camDist = w0
    return camDist

def streq(s, key):
    return isinstance(s,str) and s==key

#======================================================================
#
#  GEOMETRIC PRIMITIVES AND RENDERING CODE
#  render, sphere, cylinder, cone, billboard
#
#=====================================================================  
def render (objects, 
            elev=np.radians(30), azim=np.radians(30), camFw='auto', camUp=(0,0,1), camOri='auto', 
            camFov=np.radians(55), imageSize=(512,384), camTgt='auto', camDist='auto', zoomOut=1.0):
    '''
    See examples in p3jdemo2 for examples of render(...)
    '''
    
    if not streq(camOri, 'auto'): pass                           # if camOri was supplied explicitly, use the value
    elif not streq(camFw, 'auto'):
        camFw = normalize(camFw)
        camOri = lookAt (-camFw, camUp)  # otherwise, if camFw was supplied explicitly, use it
    else: 
        if isinstance(elev,str): elev=np.radians(eval(elev))
        if isinstance(azim,str): azim=np.radians(eval(azim))
        camOri = chooseCameraOrientation (elev, azim, camUp)     # otherwise, set camOri from elev and azim
               
    width,height = imageSize; aspect=height/width
    if streq(camTgt, 'auto'):                                    # if camTgt is auto, then set camPos and camTgt automatically
        camTgt,theDist = chooseCameraTargetAndDist (aspect, camFov, camOri, [mesh.position for mesh in objects], zoomOut=zoomOut)
        if streq(camDist, 'auto'):
            camDist = theDist
    elif streq(camDist, 'auto'):                              # if camTgt was explicit but camDist was auto, calculate 
        camDist = chooseCameraDist (aspect, camFov, camOri, [mesh.position for mesh in objects], camTgt, zoomOut=zoomOut)
    camPos = camTgt + camDist * camOri[:,2]                   # or camTgt + camOri @ [0,0,camDist]

    #lightD = p3j.DirectionalLight  (color='#FFFFFF', position=[3,5,1], intensity=1.0)
    lightD = p3j.DirectionalLight  (color='#FFFFFF', position=[2,3,1], intensity=1.0)
    lightA = p3j.AmbientLight      (color='#333333')  
    camera   = p3j.PerspectiveCamera (position=tuple(camPos), fov=np.degrees(camFov), aspect=width/height, children=[lightD], up=camUp)
    scene    = p3j.Scene (children=[*objects, camera, lightA])
    #controls = p3j.TrackballControls (controlling=camera,target=tuple(camTgt))
    controls = p3j.OrbitControls (controlling=camera,target=tuple(camTgt))   # note that this messes up camera.quaternion
    camera.quaternion = quatFromRotMat (camOri)
    renderer = p3j.Renderer (scene, camera, controls=[controls], width=width, height=height, antialias=True)
    return renderer

geomSphere = p3j.SphereBufferGeometry(1.0, 24, 24)
def sphere (position=[0,0,0], radius=1.0, color='#999999'): #, **kwargs):
    position = np.array(position)
    material = p3j.MeshPhongMaterial (color=color)
    mesh     = p3j.Mesh(geomSphere, material)
    mesh.position = tuple(position)
    mesh.scale = [radius,radius,radius]
    return mesh
def cylinder (rA, rB, radius, color='#999999', radialSegments=6):  #**kwargs):
    radius = radius * 1.0
    rA = np.array(rA)
    rB = np.array(rB)
    geometry = p3j.CylinderBufferGeometry(radius, radius, np.linalg.norm(rB-rA), radialSegments=radialSegments)
    material = p3j.MeshPhongMaterial(color=color)
    mesh     = p3j.Mesh(geometry,material)
    mesh.quaternion = quatUToV ([0,1,0], rB-rA)        # this is good
    mesh.position   = tuple((rA+rB)/2)
    return mesh
def cone (rA, rB, radius, color='#999999', radialSegments=6):  #**kwargs):
    rA = np.array(rA)
    rB = np.array(rB)
    geometry = p3j.CylinderBufferGeometry(0, radius, np.linalg.norm(rB-rA), radialSegments=radialSegments)
    material = p3j.MeshPhongMaterial(color=color)
    mesh     = p3j.Mesh(geometry,material)
    mesh.quaternion = quatUToV ([0,1,0], rB-rA)        # this is good
    mesh.position   = tuple((rA+rB)/2)
    return mesh
def arrow (posBegin, posEnd, fracShaft=.7, radShaft=.1, radHead=.2, color='#999999', radialSegments=12):  #**kwargs):
    rA = np.array(posBegin)
    rB = np.array(posEnd)
    l = np.linalg.norm (rB-rA)
    rM = rA*(1-fracShaft) + rB*(fracShaft)
    rS = radShaft*l
    rH = radHead*l
    objects = []
    objects.append ( cylinder (rA, rM, radius=rS, color=color, radialSegments=radialSegments) )
    objects.append ( cone     (rM, rB, radius=rH, color=color, radialSegments=radialSegments) )
    return p3j.Group (children=objects)
def tripod (length=4, radius=.2): # axis tripod; should build out of Arrows
    a = length; r=radius
    objects = []
    objects.append ( cylinder ([0,0,0], [a,0,0], radius=r, color='#FF0000') ) # red x axis
    objects.append ( cylinder ([0,0,0], [0,a,0], radius=r, color='#00FF00') ) # green y axis
    objects.append ( cylinder ([0,0,0], [0,0,a], radius=r, color='#0000FF') ) # blue z axis
    objects.append ( sphere ([a,0,0], radius=3*r, color='#FF0000') )
    objects.append ( sphere ([0,a,0], radius=3*r, color='#00FF00') )
    objects.append ( sphere ([0,0,a], radius=3*r, color='#0000FF') )
    return p3j.Group (children=objects)
def torus (radius, tube, color='#999999', radialSegments=12, tubularSegments=48, arc=6.283185307179586):  #**kwargs):
    geometry = p3j.TorusBufferGeometry(radius=radius, tube=tube, radialSegments=radialSegments, tubularSegments=tubularSegments, arc=arc)
    material = p3j.MeshPhongMaterial(color=color)
    mesh     = p3j.Mesh(geometry,material)
    return mesh
def box (a=1,b=1,c=1,color='#999999'):
    geometry = p3j.BoxBufferGeometry(a,b,c)
    material = p3j.MeshPhongMaterial(color=color)
    mesh     = p3j.Mesh(geometry,material)
    return mesh

#======================================================================
# txFont1 is transparent
# txFont2 has black background
#=====================================================================  
img = PIL.Image.open('monospace.png')
xijc = np.array(img)
txFont2 = p3j.DataTexture(data=xijc,                format='RGBFormat', width=1024, height=1024)  # font with black background
txFont1 = p3j.DataTexture(data=xijc[:,:,[0,0,0,0]], format='RGBAFormat', width=1024, height=1024) # font with transparent background

def billboard (string='A', position=[0,0,0], rotation=[[1,0,0],[0,1,0],[0,0,1]], fontSize=1, ha='center', va='center', fontTexture=txFont1, fontColor='#666666'):
    string = str(string)
    if string=='': return Group()
    position = np.array (position)
    rotation = np.array (rotation)
    glyphAsp = 2                  # fixed by font atlas
    glyphHei = fontSize
    glyphWid = glyphHei/glyphAsp
    a = glyphWid/2
    b = glyphHei/2
    k = .5/glyphAsp  
    material = p3j.MeshBasicMaterial(map=fontTexture, transparent=True, color=fontColor)

    objects = []
    nmax = len(string)
    for n in range(nmax):
        #======== Where to draw character
        positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,1,0],[1,0,0],[1,1,0]])*[glyphWid,glyphHei,0] + [(2*n-nmax)*a,-b,0]
        positions = position + positions @ rotation
        #======== Which part of font atlas texture to use
        c = ord(string[n])
        i = c%16 ; j = c//16 ; iB = i+.5-k ; jB = j+.98 ; iE = i+.5+k ; jE = j+.02
        uv = (np.array([[0,0],[1,0],[0,1], [0,1],[1,0],[1,1]]) * [iE-iB,jE-jB] + [iB,jB] ) / 16
        geometry = p3j.BufferGeometry(attributes=dict(
          position= p3j.BufferAttribute (np.array(positions,dtype=np.float32)),   # must cast
          uv      = p3j.BufferAttribute (np.array(uv,       dtype=np.float32)))
        )
        #======== Add the glyph!
        mesh = p3j.Mesh(geometry,material)
        objects.append (mesh)
    group = p3j.Group (children=objects)
    # group.position = tuple(position @ rotation)  # we're not using this mechanism for positioning
    return group