# from stlib.scene import Node
from stlib.components import addOrientedBoxRoi
from splib.numerics import getOrientedBoxFromTransform
from stlib.physics.rigid import RigidObject
from stlib.physics.deformable import ElasticMaterialObject
from stlib.physics.mixedmaterial import Rigidify
from sdsofa.utils.transforms import Pose, point_to_world
from sdsofa.utils.utils import parse_json
from softrobots.actuators import PullingCable

class BaseModel(object):
    def __init__(self, model_args):
        self.model_args = model_args
        self.node = model_args["node"]
        self.object = None
        self.global_translation, self.global_rotation = point_to_world(self.model_args["tf_list"])

    def create(self):
        print "initializing"

class BaseObjectModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(BaseObjectModel, self).__init__(*args, **kwargs)
        self.material_args = self.model_args["material_args"]
        self.material_args["translation"] = self.global_translation
        self.material_args["rotation"] = self.global_rotation

class BaseRigidObjectModel(BaseObjectModel):
    def __init__(self, *args, **kwargs):
        super(BaseRigidObjectModel, self).__init__(*args, **kwargs)
        self.node = RigidObject(**self.material_args)

class FloorModel(BaseRigidObjectModel):
    def __init__(self, *args, **kwargs):
        super(FloorModel, self).__init__(*args, **kwargs)

class TargetModel(BaseRigidObjectModel):
    def __init__(self, *args, **kwargs):
        super(TargetModel, self).__init__(*args, **kwargs)

class DroneModel(BaseRigidObjectModel):
    def __init__(self, *args, **kwargs):
        super(DroneModel, self).__init__(*args, **kwargs)

class BaseElasticObjectModel(BaseObjectModel):
    def __init__(self, *args, **kwargs):
        super(BaseElasticObjectModel, self).__init__(*args, **kwargs)
        self.object = ElasticMaterialObject(**self.material_args)

class FingerModel(BaseElasticObjectModel):
    def __init__(self, *args, **kwargs):
        super(FingerModel, self).__init__(*args, **kwargs)

        from stlib.physics.constraints import FixedBox

        positions = [self.global_translation[0]-50,
                     self.global_translation[1],
                     self.global_translation[2]-50,
                     self.global_translation[0]+50,
                     self.global_translation[1],
                     self.global_translation[2]+50]
        FixedBox(self.object, atPositions=positions,
                      doVisualization=True)

        # print self.global_rotation
        # addOrientedBoxRoi(self.object, name="MyBoxRoi", position=[[14.79, -1.49, 10], [14.56, -6.49, 10], [21.02, -1.77, -10]],
        #                 drawBoxes=True, scale=[100,100,100])

        


        # addOrientedBoxRoi(self.object, name="MyBoxRoi", position=[[100, 0, 10], [300, -6.49, 10], [400.02, -1.77, -10]],
        #                 drawBoxes=True, scale=[100,100,100])



        self.cables = []

        self.all_eyelet_locations = parse_json(self.model_args["eyelet_locations_file_name"])["tendons"]
        for eyelets_wrt_finger_translations in self.all_eyelet_locations:
            # for eyelet_wrt_finger_translation in eyelets_wrt_finger_translations:
            #     eyelet_wrt_world_translation = [self.global_translation[0] + eyelet_wrt_finger_translation[0],
            #                                       self.global_translation[1] + eyelet_wrt_finger_translation[1],
            #                                       self.global_translation[2] + eyelet_wrt_finger_translation[2]]
            #     myOrientedBox = getOrientedBoxFromTransform(translation=eyelet_wrt_world_translation, eulerRotation=self.global_rotation, scale=[8, 8, 40])
            #     roi = self.object.createObject("BoxROI", orientedBox=myOrientedBox, drawBoxes=True)
                # roi.init()
                # print roi.indices
                # self.rois.append(roi)
                # break

            # print eyelets_wrt_world_translations[0]
            cable = PullingCable(self.object,
                                 valueType="force",
                                 # pullPointLocation=eyelets_wrt_finger_translations[0][0],
                                 cableGeometry=eyelets_wrt_finger_translations,
                                 rotation=self.global_rotation,
                                 translation=self.global_translation)
            self.cables.append(cable)


            cable = cable.getObject("CableConstraint")
            cable.value = 0

            indices = [[157, 162, 173, 176, 177, 178, 182, 183, 185, 186, 247, 248, 249, 250, 251, 252, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 768, 773, 788, 790, 791, 792, 793, 794, 798, 799, 800, 802, 803, 804, 932, 933, 934, 935, 936, 937, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1110, 1111, 1251, 1252, 1253, 1254, 1257, 1258, 1259, 1260, 1395, 1403, 1498, 1499, 1515, 1568, 1583, 2000, 2001, 2002, 2003, 2006, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2577, 2578, 2579, 2580], [141, 146, 160, 161, 166, 167, 169, 170, 241, 242, 243, 244, 245, 246, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 533, 534, 535, 536, 539, 540, 541, 542, 748, 753, 770, 772, 778, 779, 780, 782, 783, 784, 938, 939, 940, 941, 942, 943, 1086, 1087, 1088, 1090, 1091, 1093, 1096, 1245, 1247, 1248, 1270, 1280, 1310, 1322, 1323, 1352, 1415, 1416, 1438, 1456, 1457, 1459, 1481, 1484, 1497, 1502, 1560, 1562, 1597, 1636, 1669, 1671, 1683, 1694, 1717, 2028, 2030, 2031, 2034, 2036, 2037, 2038, 2039, 2040, 2041, 2043, 2046, 2047, 2049, 2050, 2051, 2214, 2215, 2216, 2217, 2394, 2395, 2397, 2398, 2399, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2412, 2549, 2551], [125, 130, 144, 145, 150, 151, 153, 154, 235, 236, 237, 238, 239, 240, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 509, 510, 511, 512, 515, 516, 517, 518, 728, 733, 750, 752, 758, 759, 760, 762, 763, 764, 944, 945, 946, 947, 948, 949, 1072, 1073, 1074, 1077, 1079, 1080, 1082, 1230, 1232, 1233, 1287, 1330, 1331, 1359, 1362, 1396, 1437, 1449, 1494, 1500, 1513, 1565, 1587, 1661, 1675, 1693, 1710, 2052, 2053, 2054, 2056, 2057, 2060, 2062, 2063, 2065, 2068, 2069, 2071, 2072, 2073, 2074, 2075, 2218, 2219, 2220, 2221, 2373, 2374, 2376, 2377, 2378, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2391, 2534, 2562, 2564], [115, 116, 128, 129, 134, 135, 137, 138, 229, 230, 231, 232, 233, 234, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 485, 486, 487, 488, 491, 492, 493, 494, 714, 720, 730, 732, 738, 739, 740, 742, 743, 744, 950, 951, 952, 953, 954, 955, 1058, 1059, 1060, 1063, 1065, 1066, 1068, 1215, 1217, 1218, 1269, 1284, 1317, 1332, 1337, 1342, 1392, 1436, 1453, 1463, 1466, 1505, 1506, 1509, 1511, 1512, 1523, 1543, 1564, 1639, 1640, 1642, 1698, 1699, 1718, 2076, 2078, 2079, 2082, 2084, 2085, 2086, 2087, 2088, 2089, 2091, 2094, 2095, 2097, 2098, 2099, 2222, 2223, 2224, 2225, 2352, 2353, 2355, 2356, 2357, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2370, 2505, 2530, 2565], [97, 104, 110, 111, 113, 114, 121, 122, 223, 224, 225, 226, 227, 228, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 461, 462, 463, 464, 467, 468, 469, 470, 693, 701, 708, 709, 710, 712, 721, 722, 723, 724, 956, 957, 958, 959, 960, 961, 1044, 1045, 1046, 1048, 1049, 1051, 1054, 1191, 1202, 1203, 1286, 1298, 1356, 1375, 1390, 1409, 1495, 1514, 1531, 1539, 1541, 1542, 1547, 1601, 1620, 1635, 1660, 1687, 1706, 1708, 1709, 2100, 2102, 2103, 2106, 2108, 2109, 2110, 2111, 2113, 2116, 2117, 2119, 2120, 2121, 2122, 2123, 2226, 2227, 2228, 2229, 2331, 2332, 2334, 2335, 2336, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2349], [81, 88, 95, 96, 98, 101, 102, 103, 217, 218, 219, 220, 221, 222, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 437, 438, 439, 440, 443, 444, 445, 446, 673, 681, 689, 690, 691, 695, 696, 697, 698, 700, 962, 963, 964, 965, 966, 967, 1030, 1031, 1032, 1034, 1035, 1037, 1040, 1177, 1183, 1184, 1274, 1289, 1326, 1327, 1358, 1361, 1371, 1388, 1445, 1478, 1519, 1538, 1573, 1574, 1575, 1577, 1593, 1599, 1627, 1681, 1691, 2124, 2126, 2127, 2130, 2132, 2133, 2134, 2135, 2136, 2137, 2139, 2142, 2143, 2145, 2146, 2147, 2230, 2231, 2232, 2233, 2310, 2311, 2313, 2314, 2315, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2328], [65, 72, 79, 80, 82, 85, 86, 87, 211, 212, 213, 214, 215, 216, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 389, 390, 391, 392, 395, 396, 397, 398, 653, 661, 669, 670, 671, 675, 676, 677, 678, 680, 968, 969, 970, 971, 972, 973, 1002, 1003, 1004, 1007, 1009, 1010, 1012, 1162, 1168, 1169, 1276, 1288, 1333, 1334, 1374, 1377, 1380, 1381, 1383, 1471, 1501, 1516, 1548, 1550, 1563, 1571, 1572, 1643, 1674, 1688, 2148, 2150, 2151, 2154, 2156, 2157, 2158, 2159, 2160, 2161, 2163, 2166, 2167, 2169, 2170, 2171, 2234, 2235, 2236, 2237, 2268, 2269, 2271, 2272, 2273, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2286, 2506, 2517, 2569], [45, 52, 63, 64, 66, 69, 70, 71, 205, 206, 207, 208, 209, 210, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 416, 419, 420, 421, 422, 630, 638, 649, 650, 651, 655, 656, 657, 658, 660, 974, 975, 976, 977, 978, 979, 1016, 1017, 1018, 1020, 1021, 1023, 1026, 1147, 1153, 1154, 1290, 1343, 1344, 1379, 1384, 1446, 1476, 1486, 1504, 1530, 1552, 1578, 1585, 1653, 1663, 2172, 2174, 2175, 2178, 2180, 2181, 2182, 2183, 2184, 2185, 2187, 2190, 2191, 2193, 2194, 2195, 2238, 2239, 2240, 2241, 2289, 2290, 2292, 2293, 2294, 2297, 2298, 2299, 2300, 2301, 2302, 2304, 2305, 2307], [37, 38, 43, 44, 46, 49, 50, 51, 57, 59, 198, 199, 200, 201, 202, 203, 204, 256, 327, 328, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 365, 366, 367, 368, 371, 372, 373, 374, 599, 609, 610, 611, 615, 616, 617, 618, 619, 620, 621, 626, 627, 628, 632, 633, 634, 635, 637, 645, 646, 853, 854, 855, 890, 895, 896, 897, 980, 981, 982, 983, 984, 985, 988, 989, 990, 993, 995, 998, 1123, 1128, 1134, 1135, 1142, 1143, 1144, 1293, 1299, 1300, 1321, 1370, 1391, 1419, 1423, 1448, 1470, 1475, 1521, 1546, 1556, 1570, 1592, 1604, 1678, 1690, 1700, 1882, 1883, 1887, 1888, 1906, 1910, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2205, 2208, 2209, 2211, 2212, 2213, 2242, 2243, 2244, 2245, 2247, 2248, 2250, 2251, 2252, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2265, 2492, 2496, 2519, 2542, 2552], [36, 39, 41, 42, 48, 53, 54, 55, 58, 60, 264, 265, 266, 267, 268, 320, 321, 322, 325, 326, 351, 352, 353, 354, 355, 356, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 613, 614, 615, 616, 617, 621, 622, 625, 626, 628, 632, 639, 640, 641, 642, 645, 647, 853, 854, 875, 876, 877, 878, 887, 889, 891, 898, 899, 980, 981, 982, 983, 984, 985, 986, 987, 990, 994, 995, 999, 1124, 1127, 1138, 1139, 1142, 1143, 1145, 1767, 1776, 1828, 1843, 1860, 1865, 1882, 1883, 1886, 1890, 1907, 1911, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2206, 2207, 2210, 2212, 2213, 2242, 2243, 2244, 2245, 2246, 2249, 2251, 2252, 2253, 2254, 2256, 2257, 2258, 2259, 2260, 2262, 2264, 2265, 2266, 2492, 2496], [47, 56, 61, 62, 68, 73, 74, 75, 314, 315, 316, 317, 318, 319, 399, 400, 401, 402, 403, 404, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 630, 644, 648, 649, 651, 655, 662, 663, 664, 665, 974, 975, 976, 977, 978, 979, 1014, 1015, 1018, 1019, 1022, 1023, 1027, 1146, 1157, 1158, 1769, 1803, 1810, 1830, 1842, 1863, 2173, 2176, 2177, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2188, 2189, 2192, 2194, 2195, 2238, 2239, 2240, 2241, 2288, 2291, 2293, 2294, 2295, 2296, 2298, 2299, 2300, 2301, 2302, 2304, 2307, 2308], [67, 76, 77, 78, 84, 89, 90, 91, 308, 309, 310, 311, 312, 313, 375, 376, 377, 378, 379, 380, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 653, 667, 668, 669, 671, 675, 682, 683, 684, 685, 968, 969, 970, 971, 972, 973, 1000, 1001, 1004, 1008, 1009, 1011, 1013, 1161, 1172, 1173, 1748, 1756, 1786, 1787, 1816, 1829, 1831, 1832, 1845, 1852, 2149, 2152, 2153, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2164, 2165, 2168, 2170, 2171, 2234, 2235, 2236, 2237, 2267, 2270, 2272, 2273, 2274, 2275, 2277, 2278, 2279, 2280, 2281, 2283, 2285, 2286, 2287, 2506], [83, 92, 93, 94, 100, 105, 106, 107, 302, 303, 304, 305, 306, 307, 423, 424, 425, 426, 427, 428, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 673, 687, 688, 689, 691, 695, 702, 703, 704, 705, 962, 963, 964, 965, 966, 967, 1028, 1029, 1032, 1033, 1036, 1037, 1041, 1176, 1187, 1188, 1759, 1791, 1792, 1817, 1824, 1833, 1834, 1846, 1854, 2125, 2128, 2129, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2140, 2141, 2144, 2146, 2147, 2230, 2231, 2232, 2233, 2309, 2312, 2314, 2315, 2316, 2317, 2319, 2320, 2321, 2322, 2323, 2325, 2327, 2328, 2329], [99, 108, 109, 112, 117, 118, 123, 124, 296, 297, 298, 299, 300, 301, 447, 448, 449, 450, 451, 452, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 693, 707, 708, 710, 711, 716, 721, 725, 726, 727, 956, 957, 958, 959, 960, 961, 1042, 1043, 1046, 1047, 1050, 1051, 1055, 1192, 1204, 1205, 1760, 1793, 1794, 1815, 1823, 1836, 1848, 1855, 2101, 2104, 2105, 2107, 2108, 2109, 2110, 2111, 2112, 2114, 2115, 2118, 2120, 2121, 2122, 2123, 2226, 2227, 2228, 2229, 2330, 2333, 2335, 2336, 2337, 2338, 2340, 2341, 2342, 2343, 2344, 2346, 2348, 2349, 2350], [119, 120, 127, 131, 133, 136, 139, 140, 290, 291, 292, 293, 294, 295, 471, 472, 473, 474, 475, 476, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 718, 720, 730, 735, 738, 740, 741, 745, 746, 747, 950, 951, 952, 953, 954, 955, 1056, 1057, 1060, 1064, 1065, 1067, 1069, 1216, 1219, 1220, 1752, 1761, 1795, 1796, 1813, 1819, 1837, 1849, 1857, 2077, 2080, 2081, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2092, 2093, 2096, 2098, 2099, 2222, 2223, 2224, 2225, 2351, 2354, 2356, 2357, 2358, 2359, 2361, 2362, 2363, 2364, 2365, 2367, 2369, 2370, 2371], [126, 132, 143, 147, 149, 152, 155, 156, 284, 285, 286, 287, 288, 289, 495, 496, 497, 498, 499, 500, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 728, 737, 750, 755, 758, 760, 761, 765, 766, 767, 944, 945, 946, 947, 948, 949, 1070, 1071, 1074, 1078, 1079, 1081, 1083, 1231, 1234, 1235, 1757, 1797, 1808, 1814, 1838, 1850, 1856, 2052, 2053, 2055, 2058, 2059, 2061, 2062, 2063, 2064, 2066, 2067, 2070, 2072, 2073, 2074, 2075, 2218, 2219, 2220, 2221, 2372, 2375, 2377, 2378, 2379, 2380, 2382, 2383, 2384, 2385, 2386, 2388, 2390, 2391, 2392, 2562], [142, 148, 159, 163, 165, 168, 171, 172, 278, 279, 280, 281, 282, 283, 519, 520, 521, 522, 523, 524, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 748, 757, 770, 775, 778, 780, 781, 785, 786, 787, 938, 939, 940, 941, 942, 943, 1084, 1085, 1088, 1089, 1092, 1093, 1097, 1246, 1249, 1250, 1758, 1799, 1800, 1805, 1809, 1839, 1851, 1859, 2029, 2032, 2033, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2044, 2045, 2048, 2050, 2051, 2214, 2215, 2216, 2217, 2393, 2396, 2398, 2399, 2400, 2401, 2403, 2404, 2405, 2406, 2407, 2409, 2411, 2412, 2413], [158, 164, 174, 175, 179, 180, 181, 184, 187, 188, 272, 273, 274, 275, 276, 277, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 768, 777, 788, 789, 790, 795, 796, 797, 798, 800, 801, 805, 806, 807, 932, 933, 934, 935, 936, 937, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1110, 1111, 1251, 1252, 1255, 1256, 1257, 1258, 1261, 1262, 1764, 1806, 1807, 1841, 2000, 2001, 2004, 2005, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2020, 2022, 2023, 2024, 2025, 2026, 2027, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2577, 2578, 2579, 2580]]
            o = Rigidify(self.node,
                  self.object,
                  name="RigidifiedStructure",
                  groupIndices=[indices[0]])
            from splib.objectmodel import setData
            setData(o.RigidParts.dofs, showObject=True, showObjectScale=1, drawMode=2)
            setData(o.RigidParts.RigidifiedParticules.dofs, showObject=True, showObjectScale=0.1,
                    drawMode=1, showColor=[1., 1., 0., 1.])
            setData(o.DeformableParts.dofs, showObject=True, showObjectScale=.1, drawMode=2)
            o.RigidParts.createObject("FixedConstraint", indices=0)
            rootNode = self.node.getRoot()
            simulationNode = rootNode.createChild("Simulation")
            simulationNode.createObject("EulerImplicitSolver")
            simulationNode.createObject("CGLinearSolver")
            simulationNode.addChild(o)

    def create(self):
        print "hello"
        self.rois_indices = []
        for eyelets_wrt_finger_translations in self.all_eyelet_locations:
            for eyelet_wrt_finger_translation in eyelets_wrt_finger_translations:
                eyelet_wrt_world_translation = [self.global_translation[0] + eyelet_wrt_finger_translation[0],
                                                  self.global_translation[1] + eyelet_wrt_finger_translation[1],
                                                  self.global_translation[2] + eyelet_wrt_finger_translation[2]]
                myOrientedBox = getOrientedBoxFromTransform(translation=eyelet_wrt_world_translation, eulerRotation=self.global_rotation, scale=[12, 12, 40])
                roi = self.object.createObject("BoxROI", orientedBox=myOrientedBox, drawBoxes=True)
                roi.init()
                print roi.indices
                self.rois_indices.append([index for indices in roi.indices for index in indices])
                


        print self.rois_indices
        # all_roi_indices = []
        # for roi in self.rois:
        #     all_roi_indices.append(roi.indices)
        # print "indices"
        # print self.rois[0].indices
        
        # o = Rigidify(self.node,
        #           self.object,
        #           name="RigidifiedStructure",
        #           groupIndices=[[1]])
        # from splib.objectmodel import setData
        # setData(o.RigidParts.dofs, showObject=True, showObjectScale=1, drawMode=2)
        # setData(o.RigidParts.RigidifiedParticules.dofs, showObject=True, showObjectScale=0.1,
        #         drawMode=1, showColor=[1., 1., 0., 1.])
        # setData(o.DeformableParts.dofs, showObject=True, showObjectScale=0.1, drawMode=2)
        # o.RigidParts.createObject("FixedConstraint", indices=0)


    # def apply_tendon_forces(self):
    #     for cable, type, k in zip(self.cables, self.cableTypes, self.K_tendon):
    #         if type == "force":
    #             cable = cable.getObject("CableConstraint")
    #             target_length = cable.findData('cableInitialLength').value
    #             length = cable.findData('cableLength').value

    #             force = max(-k*(target_length - length), 0)
    #             cable.value = force
        
class GripperModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(GripperModel, self).__init__(*args, **kwargs)
        self.fingers = self.model_args["fingers"]

class DroneGripperModel(BaseObjectModel):
    def __init__(self, *args, **kwargs):
        self.drone_model = self.model_args["drone_model"]
        self.gripper_model = self.model_args["gripper_model"]
        self.attach_gripper()

    def attach_gripper(self):
        pass


        