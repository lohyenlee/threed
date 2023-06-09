import numpy as np
import pythreejs as p3j
import PIL.Image                     # for importing PNG files

#======================================================================
#
#  CRUCIAL HELPER FUNCTIONS
#
#=====================================================================  
def normalize (v): return v/np.linalg.norm(v)

def findNormal (u):
  if u[1]!=0: return normalize(np.cross ([1,0,0], u))
  if u[2]!=0: return normalize(np.cross ([0,1,0], u))
  if u[0]!=0: return normalize(np.cross ([0,0,1], u)) 
  raise Exception ("findNormal([0,0,0]) is illegal!")

def quatUToV (u,v): # find a quaternion that describes a rotation that rotates u to v
  u = normalize(u)
  v = normalize(v)
  udotv = u@v
  if udotv==1:  
    return (0,0,0,1)     # identity
  elif udotv==-1: 
    t = np.pi            # 180-degree rotation
    a = findNormal (u)   # any axis perpendicular to u
  else:  
    t = np.arccos(udotv)         # rotation angle
    a = normalize(np.cross(u,v)) # rotation axis
  s = np.sin(t/2); c = np.cos(t/2)
  return (s*a[0], s*a[1], s*a[2], c)  # convert axis-angle (a,t) to quaternion


#======================================================================
#
#  GEOMETRIC PRIMITIVES AND RENDERING CODE
#  render, sphere, cylinder, billboard
#
#=====================================================================  
def lookAt (forward,up):
  '''
  Return a 3x3 rotation matrix such that [0,0,1] maps to *forward* 
  and [0,1,0] maps to a direction close to *up*
  '''
  c = -normalize(forward)
  a = normalize(np.cross(up, c)) # not sure about all this
  b = normalize(np.cross(c, a))
  return np.array ([a,b,c]).T
def copysign (x, s):
  return x * np.sign(x*s)
def quatFromRotMat (m):
  w = np.sqrt( max( 0, 1 + m[0,0] + m[1,1] + m[2,2] ) ) / 2
  x = np.sqrt( max( 0, 1 + m[0,0] - m[1,1] - m[2,2] ) ) / 2
  y = np.sqrt( max( 0, 1 - m[0,0] + m[1,1] - m[2,2] ) ) / 2
  z = np.sqrt( max( 0, 1 - m[0,0] - m[1,1] + m[2,2] ) ) / 2
  x = copysign( x, m[2,1] - m[1,2] )
  y = copysign( y, m[0,2] - m[2,0] )
  z = copysign( z, m[1,0] - m[0,1] )
  return (x,y,z,w)
  
  
def chooseCameraOrientation (e=0., a=0., up=[0., 0., 1.]):
  '''
  Return a 3x3 rotation matrix corresponding to the camera orientation specified by:
  
    e = elevation in radians
    a = azimuth in radians
    up = up direction.
    
  The columns of the rotation matrix should correspond to the right, up, and back directions according to the camera.
  '''
  fw = -np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
  return lookAt (fw, up)


def chooseCameraPosition (aspect, fov, ori, points, zoomOut=1.0):
  '''
  Calculate the position (3D vector) corresponding to these camera settings:
    ori       camera orientation (3x3 matrix)
    fov       camera's vertical field of view in radians
    aspect    camera aspect ratio
    points    array of points to be captured within camera view
  '''
  unl  = points @ ori.T                  # points in camera space (for some reason this involves ori.T)
  umin,vmin,wmin = np.min (unl, axis=0)  # bounding box of 
  umax,vmax,wmax = np.max (unl, axis=0)  # point coordinates in camera space
  # How far does camera need to step back in order to capture all points?
  w0 = max(
    np.abs(umax - umin)/2/np.tan(fov/2)/aspect,
    np.abs(vmax - vmin)/2/np.tan(fov/2)
  ) 
  w0 += np.abs(wmax-wmin)/2    # step back from plane containing frontmost point
  w0 *= zoomOut          # scale by additional safety factor
  camTgt = [(umax+umin)/2, (vmax+vmin)/2, (wmax+wmin)/2] @ ori   # transform back using inverse mtrix
  camPos = camTgt + w0 * ori[:,2]                                # or center + ori @ [0,0,w0]
  #camPos = [(umax+umin)/2, (vmax+vmin)/2, (wmax+wmin)/2] @ ori + [0,0,w0]@ori   # this is incorrect!
  return camPos

def streq(s, key):
  return isinstance(s,str) and s==key
def render (objects, elev='auto', azim='auto', camOri='auto', camPos='auto', camFov=np.radians(55), camUp=[0,0,1], camTarget='auto', imageSize=[512,384], zoomOut=1.5):
  width,height = imageSize
  
  if isinstance(camTarget,str) and camTarget=='auto':
    camTarget = np.mean ([mesh.position for mesh in objects], axis=0)
  if isinstance(camOri,str) and camOri=='auto': 
    camOri = chooseCameraOrientation (e=np.radians(30), a=np.radians(30), up=camUp)
  if not streq(elev,'auto') and not streq(azim,'auto'):
    print ('Setting camOri according to elev and azim')
    camOri = chooseCameraOrientation (e=elev, a=azim, up=camUp)
  if isinstance(camPos,str) and camPos=='auto':
    camPos = chooseCameraPosition    (height/width, camFov, camOri, [mesh.position for mesh in objects], zoomOut=zoomOut)

    
  lightD = p3j.DirectionalLight  (color='#FFFFFF', position=[3,5,1], intensity=1.0)
  lightA = p3j.AmbientLight      (color='#333333')  
  camera = p3j.PerspectiveCamera (position=tuple(camPos), fov=np.degrees(camFov), aspect=width/height, children=[lightD])
  scene = p3j.Scene (children=[*objects, camera, lightA])
  camera.up = (0,0,1)
  controls = p3j.OrbitControls (controlling=camera,target=tuple(camTarget))
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
  #mesh.lookAt(tuple(rB-rA)) #mesh.rotateX(np.pi/2)  # these three.js commands don't work properly with pythreejs
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


#======================================================================
# txFont is transaprent
# txFont2 has black background
#=====================================================================  

img = PIL.Image.open('monospace.png')
xijc = np.array(img)
xijc = xijc[:,:,[0,0,0,0]]; xijc[:,:,:3]=[48,48,64];   # set font color by setting this triple
txFont = p3j.DataTexture(data=xijc, format='RGBAFormat', width=1024, height=1024)

img = PIL.Image.open('monospace.png')
xijc = np.array(img)
#xijc = xijc[:,:,[0,0,0,0]]; xijc[:,:,:3]=[99,99,99];   # set font color by setting this triple
txFont2 = p3j.DataTexture(data=xijc, format='RGBFormat', width=1024, height=1024)

def billboard (string='A', position=[0,0,0], rotation=[[1,0,0],[0,1,0],[0,0,1]], fontSize=1, ha='center', va='center', fontTexture=txFont):
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
  material = p3j.MeshBasicMaterial(map=fontTexture, transparent=True)

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
  
# def zoomOut (positions, zoomFactor, center='auto'):
#   '''
#   zoomOut(...) takes a set of positions and scales each position about a center.
#   positions:  a two-dimensional array of coordinates (e.g., 10x3)
#   zoomFactor: a factor or array of factors (e.g., [1.1, 1.1, 1.1] to zoom out by 10%)
#   center:     center; if 'auto', use center-of-mass of positions
#   '''
#   if center=='auto':
#     center = np.mean(positions,axis=0)
#   return center + (positions - center)*zoomFactor
