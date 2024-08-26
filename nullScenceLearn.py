from nuscenes import NuScenes
DATAROOT = "nuscenes_mini"
nuscenes = NuScenes('v1.0-mini',dataroot=DATAROOT)
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini',dataroot='nuscenes_mini',verbose=True)

nusc.list_scenes()

my_scene = nusc.scene[0]

first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample',first_sample_token)

sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data',my_sample['data'][sensor])

nusc.render_sample_data(cam_front_data['token'])

lidar_top_data = nusc.get('sample_data',my_sample['data']['LIDAR_TOP'])
nusc.render_sample_data(lidar_top_data['token'])

my_annotation_token = my_sample['anns'][0]
my_annotation_metadata = nusc.get('sample_annotation',my_annotation_token)

nusc.render_annotation(my_annotation_token)