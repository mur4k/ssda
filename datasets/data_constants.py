NUM_CLASSES = 19
EPS = 1e-8
CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

CITYSCAPES_CLASSES = [
	'unlabeled', 
	'ego vehicle',
	'rectification border',
	'out of roi',
	'static',
	'dynamic',
	'ground',
	'road',
	'sidewalk',
	'parking',
	'rail track',
	'building',
	'wall',
	'fence',
	'guard rail',
	'bridge',
	'tunnel',
	'pole',
	'polegroup',
	'traffic light',
	'traffic sign',
	'vegetation',
	'terrain',
	'sky',
	'person',
	'rider',
	'car',
	'truck',
	'bus',
	'caravan',
	'trailer',
	'train',
	'motorcycle',
	'bicycle',
	'license plate'
]

CITYSCAPES_LABELS = [
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	17,
	18,
	19,
	20,
	21,
	22,
	23,
	24,
	25,
	26,
	27,
	28,
	29,
	30,
	31,
	32,
	33,
	-1
]

CITYSCAPES_TRAIN = [
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	0,
	1,
	255,
	255,
	2,
	3,
	4,
	255,
	255,
	255,
	5,
	255,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	255,
	255,
	16,
	17,
	18,
	255
]

CITYSCAPES_PALETTE = [
	(0, 0, 0),
	(0, 0, 0),
	(0, 0, 0),
	(0, 0, 0),
	(0, 0, 0),
	(111, 74, 0),
	(81, 0, 81),
	(128,64,128),
	(244, 35, 232),
	(250, 170, 160),
	(230, 150, 140),
	(70, 70, 70),
	(102, 102, 156),
	(190, 153, 153),
	(180, 165, 180),
	(150, 100, 100),
	(150, 120, 90),
	(153, 153, 153),
	(153, 153, 153),
	(250, 170, 30),
	(220, 220, 0),
	(107, 142, 35),
	(152, 251, 152),
	(70, 130, 180),
	(220, 20, 60),
	(255, 0, 0),
	(0, 0, 142),
	(0, 0, 70),
	(0, 60, 100),
	(0, 0, 90),
	(0, 0, 110),
	(0, 80, 100),
	(0, 0, 230),
	(119, 11, 32),
	(0, 0, 142)
]

CITYSCAPES_LABELS2TRAIN = dict(zip(CITYSCAPES_LABELS, CITYSCAPES_TRAIN))

CITYSCAPES_LABELS2PALETTE = dict(zip(CITYSCAPES_LABELS, CITYSCAPES_PALETTE))

CITYSCAPES_MEAN = (0.2840, 0.3227, 0.2817)

CITYSCAPES_STD = (0.1858, 0.1888, 0.1855)

CITYSCAPES_SIZE = (1024, 2048)

GTA5_LABELS = [
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	17,
	18,
	19,
	20,
	21,
	22,
	23,
	24,
	25,
	26,
	27,
	28,
	29,
	30,
	31,
	32,
	33,
	34
]

GTA5_TRAIN = [
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	0,
	1,
	255,
	255,
	2,
	3,
	4,
	255,
	255,
	255,
	5,
	255,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	255,
	255,
	16,
	17,
	18,
	255
]

GTA5_LABELS2TRAIN = dict(zip(GTA5_LABELS, GTA5_TRAIN))

GTA5_LABELS2PALETTE = dict(zip(GTA5_LABELS, CITYSCAPES_PALETTE))

GTA5_MEAN = (0.4427, 0.4384, 0.4251)

GTA5_STD = (0.2614, 0.2554, 0.2499)

GTA5_SIZE = (1052, 1914)

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)