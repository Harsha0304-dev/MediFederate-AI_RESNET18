import numpy as np
from scipy import ndimage

def _shape_descriptor(width,height,depth=1.0):
    vals=sorted([float(width),float(height),float(depth)],reverse=True)
    a,b=vals[:2]; ratio=b/a if a else 0
    if ratio >= .85: return 'approximately round'
    if ratio >= .55: return 'oval/elliptical'
    return 'irregular/elongated'

def measure_mask(mask,pixel_spacing=(1.0,1.0),slice_thickness=1.0,intensity_image=None,origin=None):
    mask=np.asarray(mask)>0
    if not mask.any(): return {'detected':False,'message':'No segmented abnormal region found.'}
    labels,n=ndimage.label(mask); comps=[]
    for i,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None: continue
        component=(labels[sl]==i); ys,xs=np.where(component)
        h=(sl[0].stop-sl[0].start)*pixel_spacing[0]; w=(sl[1].stop-sl[1].start)*pixel_spacing[1]
        area=float(component.sum()*pixel_spacing[0]*pixel_spacing[1]); cy=float((ys+sl[0].start).mean()*pixel_spacing[0]); cx=float((xs+sl[1].start).mean()*pixel_spacing[1])
        item={'component':i,'location':{'x_mm':cx,'y_mm':cy},'max_dimension_mm':float(max(w,h)),
              'width_mm':float(w),'height_mm':float(h),'area_mm2':area,'pixels':int(component.sum()),
              'shape':_shape_descriptor(w,h)}
        if origin: item['approx_physical_position_mm']=[float(origin[0]+cx),float(origin[1]+cy)]
        if intensity_image is not None:
            arr=np.asarray(intensity_image); roi=arr[labels==i]
            if roi.size:
                item['mean_hu']=float(np.mean(roi)); item['min_hu']=float(np.min(roi)); item['max_hu']=float(np.max(roi)); item['median_hu']=float(np.median(roi))
        comps.append(item)
    return {'detected':True,'component_count':len(comps),'components':comps,'pixel_spacing_mm':[float(pixel_spacing[0]),float(pixel_spacing[1])],
            'slice_thickness_mm':float(slice_thickness),'note':'Accurate physical measurements require correct DICOM/NIfTI spacing and a validated segmentation mask.'}

def measure_volume(mask3d,spacing=(1,1,1)):
    vox=np.asarray(mask3d)>0; v=float(vox.sum()*np.prod(spacing))
    return {'volume_mm3':v,'volume_cm3':v/1000,'voxel_spacing_mm':list(map(float,spacing))}
