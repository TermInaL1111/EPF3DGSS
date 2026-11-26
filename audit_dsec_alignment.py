"""Audit REALM Warper's rotation-only approximation on native DSEC cameras.

Warper: REALM/realm/realm/utils/transforms.py::_build_maps.
DSEC schema adaptation is ours; sparse disparity is used only to measure the
translation/parallax omitted by this approximation, not as student input.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from dsec_data import DSECSequence,camera_matrix,lower_bound,realm_rotation_grid


def main(a):
    a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for seq in a.sequences:
        d=DSECSequence(a.root,seq);cal=d.calib;grid=realm_rotation_grid(cal)
        try:
            for index in np.array(d.indices)[np.linspace(0,len(d.indices)-1,a.samples,dtype=int)]:
                im=cv2.imread(str(d.root/f'images/left/rectified/{index:06}.png'))
                disp=cv2.imread(str(d.root/f'disparity/image/{index:06}.png'),-1).astype(np.float32)/256
                xyz=cv2.reprojectImageTo3D(disp,np.array(cal['disparity_to_depth']['cams_12']))
                valid=(disp>0)&np.isfinite(xyz).all(-1)&(xyz[...,2]>0)
                y,x=np.nonzero(valid);points=xyz[valid]
                r=np.array(cal['extrinsics']['R_rect1']);t=np.array(cal['extrinsics']['T_10'])
                local=(points@r-t[:3,3])@t[:3,:3]
                ev=cal['intrinsics']['cam0'];ke=camera_matrix(ev['camera_matrix'])
                exact,_=cv2.projectPoints(local,np.zeros(3),np.zeros(3),ke,np.array(ev['distortion_coeffs']))
                exact=exact[:,0];keep=(local[:,2]>0)&(exact[:,0]>=0)&(exact[:,0]<639)&(exact[:,1]>=0)&(exact[:,1]<479)
                approx=np.concatenate([cv2.remap(grid,chunk[:,0,None],chunk[:,1,None],cv2.INTER_LINEAR)[:,0] for chunk in np.array_split(exact[keep].astype(np.float32),max(1,(keep.sum()+9999)//10000))])
                err=np.linalg.norm(approx-np.stack([x[keep],y[keep]],-1),axis=1)
                rows.append({'sequence':seq,'frame':int(index),'points':len(err),
                    'rgb_pixel_error_p50_p95_max':np.percentile(err,[50,95,100]).tolist(),
                    'fraction_error_over_14px':float((err>14).mean())})
                warped=cv2.remap(im,grid,None,cv2.INTER_LINEAR)
                f=d.files[0];end=int(d.times[index])-int(f['t_offset'][()]);hi=lower_bound(f['events/t'],end);lo=max(0,hi-150000)
                ex,ey=f['events/x'][lo:hi],f['events/y'][lo:hi]
                canvas=warped.copy();canvas[ey,ex]=(.5*canvas[ey,ex]+np.array([0,0,127])).astype(np.uint8)
                cv2.imwrite(str(a.out/f'{seq}_{index:06}.jpg'),np.concatenate([warped,canvas],1))
        finally:d.close()
    (a.out/'alignment.json').write_text(json.dumps({'approximation':'REALM rotation-only warper adapted to DSEC raw event / rectified image calibration','rows':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--sequences',nargs='+',required=True);p.add_argument('--samples',type=int,default=4)
    main(p.parse_args())
