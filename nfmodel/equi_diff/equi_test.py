import numpy as np
import time
import torch
import absl.flags as flags
FLAGS = flags.FLAGS
from absl import app

def eval_group(argv):

    R60=get_anchorsV()
    R12=get_anchorsV12()
    index=21
    index2=45
    R=torch.from_numpy(R60[index]).float()
    rotation_dict=torch.load(FLAGS.rotation_path)
    vs_=rotation_dict['vs'].float()
    vs=vs_
    faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),(1,2,4),(2,7,8),(3,7,9),(3,5,11),(4,5,6),(2,4,10),(2,8,10),
           (7,8,9),(3,9,11),(5,6,11),(4,6,10),(0,8,10),(0,6,10),(0,6,11),(0,9,11),(0,8,9)]
    faces=torch.from_numpy(np.array(faces))
    color2face=torch.from_numpy(np.array([(0,8,12,15),(1,9,13,19),(2,5,14,18),(3,6,10,17),(4,7,11,16)]))
    face_normal=vs[faces,:].sum(1)
    face_normal=F.normalize(face_normal,dim=-1)

    face2colors=np.array([(1,2,4),(0,2,3),(1,3,4),(0,2,4),(0,1,3),(0,3,4),(0,1,4),(0,1,2),(1,2,3),
                          (2,3,4),(0,1,2),(1,2,3),(2,3,4),(0,3,4),(0,1,4),(1,3,4),(0,2,3),(1,2,4),(0,1,3),(0,2,4)])
    face2colors=torch.from_numpy(face2colors)
    roll=np.array([(0,1,2,3,4),(1,2,3,4,0),(2,3,4,0,1),(3,4,0,1,2),(4,0,1,2,3)])
    roll=torch.from_numpy(roll)


    vert2colors=np.array([(0,4,3,2,1),(0,1,2,3,4),(0,4,1,3,2),(0,4,2,1,3),(0,2,1,4,3),(0,3,2,4,1),(0,1,3,4,2),
                          (0,2,4,3,1),(0,1,4,2,3),(0,3,4,1,2),(0,3,1,2,4),(0,2,3,1,4)])
    vert2colors=torch.from_numpy(vert2colors)

    weight=torch.tensor([2.0,4.0,6.0,8.0,10.0])/2

    vertices_1 = torch.randn(1,1, 3)
    vertices_1[0,0,:]=torch.tensor([0,0.01,1])
    vertices_2= torch.einsum('ij,bpj->bpi',R,vertices_1)

    vert_normal_1=torch.einsum('f i , b q i -> b q f',face_normal,vertices_1)
    vert_normal_1=torch.tanh(vert_normal_1)
    vert_color_1=vert_normal_1[:,:,color2face].sum(-1)

    vert_weight_1=vert_color_1[:,:,vert2colors]
    vert_weight_1_roll=(vert_weight_1[:,:,:,roll]*weight).sum(-1)
    vert_weight_1=torch.max(vert_weight_1_roll,dim=-1)[0]
    vert_value_1=torch.einsum('b d v , v i-> b d i',vert_weight_1,vs)


    vert_normal_2=torch.einsum('f i , b q i -> b q f',face_normal,vertices_2)
    vert_normal_2=torch.tanh(vert_normal_2)
    vert_color_2=vert_normal_2[:,:,color2face].sum(-1)

    vert_weight_2=vert_color_2[:,:,vert2colors]
    vert_weight_2_roll=(vert_weight_2[:,:,:,roll]*weight).sum(-1)
    vert_weight_2=torch.max(vert_weight_2_roll,dim=-1)[0]
    vert_value_2=torch.einsum('b d v , v i-> b d i',vert_weight_2,vs)

    vert_value_1_ro=torch.einsum('ij,bpj->bpi',R,vert_value_1)
    print(1)


def eval_group_2(argv):

    R60=get_anchorsV()
    R12=get_anchorsV12()
    index=21
    index2=45
    R=torch.from_numpy(R12[4]).float()
    rotation_dict=torch.load(FLAGS.rotation_path)
    vs_=rotation_dict['vs'].float()
    faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
           (1,2,4),(2,7,8),(3,7,9),(3,5,11),
           (4,5,6),(2,4,10),(2,8,10),(7,8,9),
           (3,9,11),(5,6,11),(4,6,10),(0,8,10),
           (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

    face_normal=vs[faces,:].sum(1)
    face_normal=torch.from_numpy(face_normal).float()
    face_normal=F.normalize(face_normal,dim=-1)

    faces=torch.from_numpy(np.array(faces))

    cube_to_face_normal=[(1,4,7,8,11,10,16,18),
                         (0,2,11,12,8,9,15,17),
                         (1,3,12,13,5,9,19,16),
                         (4,2,5,6,14,13,15,18),
                         (0,3,6,7,10,14,19,17),]

    cube_to_face_normal_orth=[(1,11,7,4),
                              (0,11,2,9),
                              (1,12,5,3),
                              (4,2,5,14),
                              (0,10,6,3),]
    cube_to_face_normal_orth=torch.from_numpy(np.array(cube_to_face_normal_orth))

    face_to_cube=[(1,4),(2,0),(3,1),(4,2),
                  (0,3),(3,2),(4,3),(0,4),
                  (1,0),(2,1),(4,0),(0,1),
                  (1,2),(2,3),(3,4),(1,3),
                  (0,2),(4,1),(3,0),(2,4)]
    face_to_cube=torch.from_numpy(np.array(face_to_cube))

    cube_to_face_normal=torch.from_numpy(np.array(cube_to_face_normal))

    cubes=face_normal[cube_to_face_normal]

    cubs_orth=face_normal[cube_to_face_normal_orth]
    new_cubs_orth=torch.zeros_like(cubs_orth)[:,:3]
    new_cubs_orth[:,0]=cubs_orth[:,1]-cubs_orth[:,0]
    new_cubs_orth[:,1]=cubs_orth[:,2]-cubs_orth[:,0]
    new_cubs_orth[:,2]=cubs_orth[:,3]-cubs_orth[:,0]
    cubs_orth=torch.cat([new_cubs_orth,-new_cubs_orth],dim=1)
    cubs_orth=F.normalize(cubs_orth,dim=-1)

    # show_open3d(cubs_orth[0],cubs_orth[0][1:4]+1)

    vertices_1 = torch.randn(1,1, 3)
    vertices_1[0,0,:]=torch.tensor([0,0.01,1])
    vertices_2= torch.einsum('ij,bpj->bpi',R,vertices_1)



    vert_normal_1=torch.einsum('c n i , b q i -> b q c n',cubs_orth,vertices_1)
    vert_normal_1=vert_normal_1.max(-1)[0]

    vert_value_1=vert_normal_1[:,:,face_to_cube[:,0]]*(vert_normal_1[:,:,face_to_cube[:,0]]>vert_normal_1[:,:,face_to_cube[:,1]])
    vert_value_1=torch.einsum('b q n , n i->b q i',vert_value_1,face_normal)

    vert_normal_2=torch.einsum('c n i , b q i -> b q c n',cubs_orth,vertices_2)
    vert_normal_2=vert_normal_2.max(-1)[0]

    vert_value_2=vert_normal_2[:,:,face_to_cube[:,0]]*(vert_normal_2[:,:,face_to_cube[:,0]]>vert_normal_2[:,:,face_to_cube[:,1]])
    vert_value_2=torch.einsum('b q n , n i->b q i',vert_value_2,face_normal)




    vert_value_1_ro=torch.einsum('ij,bpj->bpi',R,vert_value_1)
    print(1)


def eval_group_3(argv):

    R60=get_anchorsV()
    R12=get_anchorsV12()
    index=21
    index2=45
    R=torch.from_numpy(R60[41]).float()
    rotation_dict=torch.load(FLAGS.rotation_path)
    vs_=rotation_dict['vs'].float()



    v2colors=[0,0,2,5,3,4,1,1,4,3,5,2]
    color2v=[(0,1),(6,7),(2,11),(4,9),(5,8),(3,10)]
    color_com=[(0,1,2,3,4,5),
               (0,5,4,3,2,1),
               (2,5,4,1,0,3),
               (5,0,1,3,2,4),
               (3,5,2,0,4,1),
               (4,0,5,2,1,3),
               (1,3,4,2,0,5),
               (1,0,2,4,3,5),
               (4,1,2,5,0,3),
               (3,1,4,0,2,5),
               (5,1,0,4,2,3),
               (2,4,5,3,0,1)]

    v2colors=torch.from_numpy(np.array(v2colors))
    color2v=torch.from_numpy(np.array(color2v))
    color_com=torch.from_numpy(np.array(color_com))



    roll=np.array([(0,1,2,3,4),(1,2,3,4,0),(2,3,4,0,1),(3,4,0,1,2),(4,0,1,2,3)])
    roll=torch.from_numpy(roll)

    weight=torch.from_numpy(np.array([4e10,4e8,4e6,4e4,4e2]))

    vertices_1 = torch.randn(1,1, 3)
    vertices_1[0,0,:]=torch.tensor([0,0.001,1])
    vertices_2= torch.einsum('ij,bpj->bpi',R,vertices_1)



    vert_normal_1=torch.einsum('n i , b q i -> b q  n',vs_,vertices_1)
    color_1=vert_normal_1[:,:,color2v].max(-1)[0]

    vert_color_1=color_1[:,:,color_com][:,:,:,0]
    vert_color_1=torch.einsum('bqv, vi -> bqvi',vert_color_1,vs_)
    vert_color_pair_1=vert_color_1[:,:,color2v,:]
    vert_color_roll_1=color_1[:,:,color_com][:,:,:,1:]

    vert_color_roll_1=vert_color_roll_1[:,:,:,roll]*weight
    vert_color_roll_1=vert_color_roll_1.sum(-1).max(-1)[0]

    vert_color_pair_index_1=torch.max(vert_color_roll_1[:,:,color2v],dim=-1,keepdim=True)[1]
    vert_value_1=torch.gather(vert_color_pair_1,3,vert_color_pair_index_1[:,:,:,:,None].repeat(1,1,1,1,3)).sum(-2).sum(-2)


    vert_normal_2=torch.einsum('n i , b q i -> b q  n',vs_,vertices_2)
    color_2=vert_normal_2[:,:,color2v].max(-1)[0]

    vert_color_2=color_2[:,:,color_com][:,:,:,0]
    vert_color_2=torch.einsum('bqv, vi -> bqvi',vert_color_2,vs_)
    vert_color_pair_2=vert_color_2[:,:,color2v,:]
    vert_color_roll_2=color_2[:,:,color_com][:,:,:,1:]

    vert_color_roll_2=vert_color_roll_2[:,:,:,roll]*weight
    vert_color_roll_2=vert_color_roll_2.sum(-1).max(-1)[0]

    vert_color_pair_index_2=torch.max(vert_color_roll_2[:,:,color2v],dim=-1,keepdim=True)[1]
    vert_value_2=torch.gather(vert_color_pair_2,3,vert_color_pair_index_2[:,:,:,:,None].repeat(1,1,1,1,3)).sum(-2).sum(-2)



    vert_value_1_ro=torch.einsum('ij,bpj->bpi',R,vert_value_1)
    print(1)

def eval_group_4(argv):

    R60=get_anchorsV()
    R12=get_anchorsV12()
    index=21
    index2=45
    R=torch.from_numpy(R60[index]).float()
    rotation_dict=torch.load(FLAGS.rotation_path)
    vs_=rotation_dict['vs'].float()
    vs=vs_
    faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),(1,2,4),(2,7,8),(3,7,9),(3,5,11),(4,5,6),(2,4,10),(2,8,10),
           (7,8,9),(3,9,11),(5,6,11),(4,6,10),(0,8,10),(0,6,10),(0,6,11),(0,9,11),(0,8,9)]
    faces=torch.from_numpy(np.array(faces))
    color2face=torch.from_numpy(np.array([(0,8,12,15),(1,9,13,19),(2,5,14,18),(3,6,10,17),(4,7,11,16)]))
    face_normal=vs[faces,:].sum(1)
    face_normal=F.normalize(face_normal,dim=-1)

    face2colors=np.array([(1,2,4),(0,2,3),(1,3,4),(0,2,4),(0,1,3),(0,3,4),(0,1,4),(0,1,2),(1,2,3),
                          (2,3,4),(0,1,2),(1,2,3),(2,3,4),(0,3,4),(0,1,4),(1,3,4),(0,2,3),(1,2,4),(0,1,3),(0,2,4)])
    face2colors=torch.from_numpy(face2colors)
    roll=np.array([(0,1,2,3,4),(1,2,3,4,0),(2,3,4,0,1),(3,4,0,1,2),(4,0,1,2,3)])
    roll=torch.from_numpy(roll)


    vert2colors=np.array([(0,4,3,2,1),(0,1,2,3,4),(0,4,1,3,2),(0,4,2,1,3),(0,2,1,4,3),(0,3,2,4,1),(0,1,3,4,2),
                          (0,2,4,3,1),(0,1,4,2,3),(0,3,4,1,2),(0,3,1,2,4),(0,2,3,1,4)])
    vert2colors=torch.from_numpy(vert2colors)

    vert2matrix=torch.zeros(12,5,5).long()
    for i in range(12):
        for j in range(5):
            pre=vert2colors[i][j]
            post=vert2colors[i][(j+1)%5]
            vert2matrix[i][post][pre]=1



    weight=torch.tensor([2.0,4.0,6.0,8.0,10.0])

    vert2weight=torch.einsum('v i j, j-> vi',vert2matrix.float(),weight)

    vertices_1 = torch.randn(1,1, 3)
    vertices_1[0,0,:]=torch.tensor([0,0.02,1])
    vertices_2= torch.einsum('ij,bpj->bpi',R,vertices_1)

    vert_normal_1=torch.einsum('v i , b q i -> b q v',vs,vertices_1)

    vert_color_1=torch.einsum(' v c , b q v -> b q c',vert2weight,vert_normal_1)

    vert_normal_2=torch.einsum('v i , b q i -> b q v',vs,vertices_2)

    vert_color_2=torch.einsum(' v c , b q v -> b q c',vert2weight,vert_normal_2)




    vert_value_1_ro=torch.einsum('ij,bpj->bpi',R,vert_value_1)
    print(1)


def eval_group_5(argv):

    R60=get_anchorsV()
    R12=get_anchorsV12()
    index=21
    index2=45
    R=torch.from_numpy(R60[index]).float()
    rotation_dict=torch.load(FLAGS.rotation_path)
    vs_=rotation_dict['vs'].float()
    vs=vs_
    faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),(1,2,4),(2,7,8),(3,7,9),(3,5,11),(4,5,6),(2,4,10),(2,8,10),
           (7,8,9),(3,9,11),(5,6,11),(4,6,10),(0,8,10),(0,6,10),(0,6,11),(0,9,11),(0,8,9)]
    faces=torch.from_numpy(np.array(faces))
    color2face=torch.from_numpy(np.array([(0,8,12,15),(1,9,13,19),(2,5,14,18),(3,6,10,17),(4,7,11,16)]))
    face_normal=vs[faces,:].sum(1)
    face_normal=F.normalize(face_normal,dim=-1)

    face2colors=np.array([(1,2,4),(0,2,3),(1,3,4),(0,2,4),(0,1,3),(0,3,4),(0,1,4),(0,1,2),(1,2,3),
                          (2,3,4),(0,1,2),(1,2,3),(2,3,4),(0,3,4),(0,1,4),(1,3,4),(0,2,3),(1,2,4),(0,1,3),(0,2,4)])
    face2colors=torch.from_numpy(face2colors)
    roll=np.array([(0,1,2,3,4),(1,2,3,4,0),(2,3,4,0,1),(3,4,0,1,2),(4,0,1,2,3)])
    roll=torch.from_numpy(roll)


    vert2colors=np.array([(0,4,3,2,1),(0,1,2,3,4),(0,4,1,3,2),(0,4,2,1,3),(0,2,1,4,3),(0,3,2,4,1),(0,1,3,4,2),
                          (0,2,4,3,1),(0,1,4,2,3),(0,3,4,1,2),(0,3,1,2,4),(0,2,3,1,4)])
    vert2colors=torch.from_numpy(vert2colors)

    face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                  (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                  (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                  (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                  (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
    face_to_cube=torch.from_numpy(np.array(face_to_cube))

    face_to_cube_2=face_to_cube[:,:2]
    face_to_cube_1=face_to_cube[:,0]




    weight=torch.tensor([100,1.0])



    vertices_1 = torch.randn(1,1, 3)
    vertices_1[0,0,:]=torch.tensor([0,0.2,1])
    vertices_2= torch.einsum('ij,bpj->bpi',R,vertices_1)

    vert_normal_1=torch.einsum('f i , b q i -> b q f',face_normal,vertices_1)

    vert_color_weight_1=torch.tanh(vert_normal_1[:,:,:])
    vert_color_1=torch.zeros_like(vert_normal_1)[:,:,:1].repeat(1,1,5)

    vert_color_1.scatter_add_(-1,face_to_cube_1.reshape(-1)[None,None,:],vert_color_weight_1.reshape(1,1,-1))

    vert_value_1=vert_color_1[:,:,face_to_cube[:,0]]*(vert_color_1[:,:,face_to_cube[:,0]]>vert_color_1[:,:,face_to_cube[:,1]])
    vert_value_1=torch.einsum('b q n , n i->b q i',vert_value_1,face_normal)


    vert_normal_2=torch.einsum('f i , b q i -> b q f',face_normal,vertices_2)

    vert_color_weight_2=torch.tanh(vert_normal_2[:,:,:])
    vert_color_2=torch.zeros_like(vert_normal_2)[:,:,:1].repeat(1,1,5)

    vert_color_2.scatter_add_(-1,face_to_cube_1.reshape(-1)[None,None,:],vert_color_weight_2.reshape(1,1,-1))

    vert_value_2=vert_color_2[:,:,face_to_cube[:,0]]*(vert_color_2[:,:,face_to_cube[:,0]]>vert_color_2[:,:,face_to_cube[:,1]])
    vert_value_2=torch.einsum('b q n , n i->b q i',vert_value_2,face_normal)



    vert_value_1_ro=torch.einsum('ij,bpj->bpi',R,vert_value_1)
    print(1)



if __name__ == "__main__":
    from nfmodel.equ_gcn3d import *
    from config.equi_diff.config import *
    app.run(eval_group_5)