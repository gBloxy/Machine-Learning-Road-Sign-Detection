
from sys import argv
from lxml import etree
from PIL import Image

# Get road sign center position
def get_position(obj):
    bbox = obj.find("bndbox")

    xmin = int(bbox.findtext("xmin"))
    ymin = int(bbox.findtext("ymin"))
    xmax = int(bbox.findtext("xmax"))
    ymax = int(bbox.findtext("ymax"))

    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    return center_y, center_x

anno = './road-sign-detection/annotations/'
imgs = './road-sign-detection/images/'

# iterate over all annotations
iterations = 1
for i in range(877):
    filepath = anno + 'road' + str(i) + '.xml'
    tree = etree.parse(filepath)
    root = tree.getroot()
    objects = root.xpath("./object[name='speedlimit']")

    # check if there is a speed limit sign
    if not len(objects):
        continue
    
    # check if image signs already contain a speed label
    done = False
    for o in objects:
        if o.find('speed') is not None:
            done = True
            break
    if done:
        continue

    print('Road', i, '-', str(iterations), ': ', end='')
    
    # display the image
    imagepath = imgs + 'road' + str(i) + '.png'
    img = Image.open(imagepath)
    img.show()

    # assure signs in diagonal order
    if len(objects) > 1:
        x0, y0 = get_position(objects[0])
        x1, y1 = get_position(objects[1])
        if (x1 - x0) + (y1 - y0) < 0:
            objects = (objects[1], objects[0])
        print('[Two limits founds] ', end='')

    # iterate over speed limit signs
    for o in objects:
        # get a correct label from input
        label = input()
        while not label.isdigit():
            print('label need to be a digit')
            label = input()
        
        # add label to tree
        etree.SubElement(o, "speed").text = label

    tree.write(
        filepath,
        encoding="UTF-8",
        pretty_print=True
    )
    iterations += 1

