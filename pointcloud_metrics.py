"""Local bidirectional point-cloud metrics on matched, Sim(3)-aligned samples."""

import numpy as np
from scipy.spatial import cKDTree


def normals(points, tree, neighbors):
    ids = tree.query(points,k=neighbors)[1]
    neighborhoods = points[ids]
    centered = neighborhoods-neighborhoods.mean(axis=1,keepdims=True)
    covariance = centered.transpose(0,2,1)@centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[...,0], eigenvalues[:,1] > 1e-12


def pointcloud_scores(aligned_points, reference_points, limit=8192, neighbors=16):
    """Distances in reference units (metres), with unsigned normal agreement."""
    if aligned_points.shape != reference_points.shape or len(aligned_points) < neighbors:
        raise ValueError('Point-cloud metrics require matched support and enough neighbors')
    sample = np.linspace(0,len(aligned_points)-1,min(limit,len(aligned_points)),dtype=int)
    estimate, reference = (np.asarray(points[sample],dtype=np.float64)
                           for points in (aligned_points,reference_points))
    pred_tree, ref_tree = cKDTree(estimate), cKDTree(reference)
    acc, nearest_ref = ref_tree.query(estimate)
    comp, nearest_pred = pred_tree.query(reference)
    norm_e, valid_e = normals(estimate,pred_tree,neighbors)
    norm_r, valid_r = normals(reference,ref_tree,neighbors)
    ref_ok = valid_e & valid_r[nearest_ref]
    pred_ok = valid_r & valid_e[nearest_pred]
    to_reference = np.abs((norm_e[ref_ok]*norm_r[nearest_ref[ref_ok]]).sum(-1))
    to_estimate = np.abs((norm_r[pred_ok]*norm_e[nearest_pred[pred_ok]]).sum(-1))
    nc = (float(np.clip((to_reference.mean()+to_estimate.mean())/2,0.,1.))
          if len(to_reference) and len(to_estimate) else None)
    return dict(cloud_accuracy_m=float(acc.mean()),cloud_completeness_m=float(comp.mean()),
                cloud_normal_consistency=nc, cloud_points=len(sample),
                cloud_valid_normal_fraction=float((ref_ok.mean()+pred_ok.mean())/2))
