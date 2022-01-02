MINI_LABEL_MAP = {'Fork': 0, 'Radio': 1, 'Glasses': 2, 'Webcam': 3, 'Speaker': 4, 'Keyboard': 5, 'Sneakers': 6, 'Bucket': 7, 'Alarm_Clock': 8, 'Exit_Sign': 9, 'Calculator': 10, 'Folder': 11, 'Lamp_Shade': 12, 'Refrigerator': 13, 'Pen': 14, 'Soda': 15, 'TV': 16, 'Candles': 17, 'Chair': 18, 'Computer': 19, 'Kettle': 20, 'Monitor': 21, 'Marker': 22, 'Scissors': 23, 'Couch': 24, 'Trash_Can': 25, 'Ruler': 26, 'Telephone': 27, 'Hammer': 28, 'Helmet': 29, 'ToothBrush': 30, 'Fan': 31, 'Spoon': 32, 'Calendar': 33, 'Oven': 34, 'Eraser': 35, 'Postit_Notes': 36, 'Mop': 37, 'Table': 38, 'Laptop': 39, 'Pan': 40, 'Bike': 41, 'Clipboards': 42, 'Shelf': 43, 'Paper_Clip': 44, 'File_Cabinet': 45, 'Push_Pin': 46, 'Mug': 47, 'Bottle': 48, 'Knives': 49, 'Curtains': 50, 'Printer': 51, 'Drill': 52, 'Toys': 53, 'Mouse': 54, 'Flowers': 55, 'Desk_Lamp': 56, 'Pencil': 57, 'Sink': 58, 'Batteries': 59, 'Bed': 60, 'Screwdriver': 61, 'Backpack': 62, 'Flipflops': 63, 'Notebook': 64}
MINI_LABEL_MAP_INV = {v: k for k, v in MINI_LABEL_MAP.items()}


# d = {}

# i = 0
# with open("../hw4_data/office/val.csv", 'r') as f:
#     for l in f.readlines():
#         label = l.split(',')[-1].split('\n')[0]
#         print(l)
#         if label == "label":
#             continue
#         try:
#             d[label]
#         except KeyError as e:
#             d[label] = i
#             i += 1
# print(d)