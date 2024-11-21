import geopandas as gpd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# CPA (Closest point of approach) 
# TCPA (Time to CPA) Estiamted time until target reaches CPA
# DCPA (Distance at CPA) Distance between own ship and CPA  

NMI_IN_KM = 1.852 # 1.852 is the length of a nautical mile in km
EPS = 1e-9 # Epsilon value added to avoid division by zero

def calc_cpa(data): 
    lon_delta = data['vessel_1_longitude'] - data['vessel_2_longitude']
    lat_delta = data['vessel_1_latitude'] - data['vessel_2_latitude']

    vessel_1_xy = np.array([data['vessel_1_longitude'], data['vessel_1_latitude']]).reshape(1, -1)
    vessel_2_xy = np.array([data['vessel_2_longitude'], data['vessel_2_latitude']]).reshape(1, -1)

    euclidian_dist = haversine_distances(vessel_1_xy, vessel_2_xy) / NMI_IN_KM

    rel_speed_x, rel_speed_y, rel_speed_mag = calc_rel_speed(data["vessel_1_speed"], data["vessel_1_course_rad"],
                                                             data["vessel_2_speed"], data["vessel_2_course_rad"])
   
    rel_movement_direction = np.arctan2(rel_speed_x, rel_speed_y)
    azimuth_target_to_own = np.arctan2(lon_delta, lat_delta)

    rel_bearing = azimuth_target_to_own - data["vessel_1_course_rad"]

    CPA_angle = rel_movement_direction - azimuth_target_to_own - np.pi
    dcpa = euclidian_dist * np.sin(CPA_angle)
    tcpa = euclidian_dist * np.cos(CPA_angle) / rel_speed_mag

    return {
            "euclidian_dist": euclidian_dist,
            "rel_speed_mag": rel_speed_mag,
            "rel_movement_direction": normalize_angle(rel_movement_direction),
            "azimuth_target_to_own": normalize_angle(azimuth_target_to_own),
            "rel_bearing": normalize_angle(rel_bearing),
            "dcpa": dcpa,
            "tcpa": tcpa
        }

def calc_cri(data, euclidian_dist, rel_movement_direction, azimuth_target_to_own, rel_bearing, dcpa, tcpa, rel_speed_mag, weights=[0.4457, 0.2258, 0.1408, 0.1321, 0.0556]):
    result = np.nan
    
    d1, d2 = calc_safety_domain(azimuth_target_to_own)
    u_dcpa = cpa_membership(np.abs(dcpa), d1, d2)

    t1, t2 = calc_collision_eta(np.abs(dcpa), rel_speed_mag, d1, d2)
    u_tcpa = cpa_membership(np.abs(tcpa), t1, t2)

    crit_safe_dist, avoidance_measure_dist = calc_crit_dist(data["vessel_1_length"] / NMI_IN_KM*1000, rel_bearing) # Length is in meters, so we convert it to nmi
    u_dist = cpa_membership(euclidian_dist, crit_safe_dist, avoidance_measure_dist)

    u_bearing = rel_bearing_membership(rel_bearing)

    u_speed = speed_ratio_membership(data["vessel_1_speed"], data["vessel_2_speed"], rel_movement_direction)

    result = np.dot(weights, [u_dcpa, u_tcpa, u_dist, u_bearing, u_speed])

    return result

def cpa_membership(value, min, max):
    if value <= min:
        return 1
    elif min < value <= max:
        return ((max - value) / (max - min)) ** 2
    else:
        return 0

def calc_collision_eta(dcpa, rel_speed, d1, d2):
    if dcpa <= d1:
        t1 = np.sqrt(d1 ** 2 - dcpa ** 2) / rel_speed
    else:
        t1 = (d1 - dcpa) / rel_speed

    t2 = np.sqrt(d2 ** 2 - dcpa ** 2) / rel_speed

    return t1, t2

def calc_crit_dist(own_length, rel_bearing):
    crit_safe_dist = own_length * 12
    angle = rel_bearing - np.deg2rad(19)
    avoidance_measure_dist = 1.7 * np.cos(angle) + np.sqrt(4.4 + 2.89 * np.cos(angle) ** 2)

    return crit_safe_dist, avoidance_measure_dist

def rel_bearing_membership(rel_bearing):
    angle = rel_bearing - np.deg2rad(19)
    result = 1/2 * (np.cos(angle) + np.sqrt(440/289 + np.cos(angle) ** 2)) - 5/17
    return result

def speed_ratio_membership(own_speed, target_speed, rel_course):
    speed_ratio = target_speed / own_speed

    denom = speed_ratio * np.sqrt(speed_ratio ** 2 + 1 + 2 * speed_ratio * np.sin(rel_course)) + EPS
    assert denom > 0, ValueError(f'Division by zero {speed_ratio=}, {rel_course=}, {denom=}')
    
    return 1/(1 + 2/denom)

# def calc_safety_domain(azimuth_angle):
#     azimuth_angle_deg = np.degrees(azimuth_angle)

#     d1 = np.nan  

#     if (355 <= azimuth_angle_deg <= 360) or (0 <= azimuth_angle_deg <= 67.5):
#         d1 = 1.1
#     elif 67.5 < azimuth_angle_deg <= 112.5:
#         d1 = 1.0
#     elif 112.5 < azimuth_angle_deg <= 247.5:
#         d1 = 0.6
#     elif 247.5 < azimuth_angle_deg <= 355:
#         d1 = 0.9

#     return d1, 2 * d1

def calc_safety_domain(azimuth_angle):
    d1 = np.nan

    angles = np.array([0, 5*np.pi/8, np.pi, 11*np.pi/8, 2*np.pi])     

    if angles[0] <= azimuth_angle < angles[1]:
        d1 = 1.1 - 0.2 * azimuth_angle / np.pi

    elif angles[1] <= azimuth_angle < angles[2]:
        d1 = 1.0 - 0.4 * azimuth_angle / np.pi

    elif angles[2] <= azimuth_angle < angles[3]:
        d1 = 1.0 - 0.4 * (2 * np.pi - azimuth_angle) / np.pi

    elif angles[3] <= azimuth_angle < angles[4]:
        d1 = 1.1 - 0.4 * (2 * np.pi - azimuth_angle) / np.pi

    return d1, 2 * d1

def normalize_angle(angle):
    return angle % (2*np.pi)


def calc_rel_speed(own_speed, own_course, target_speed, target_course):
    # Course should be in radians. If we are larger than 2 * pi we assume we are in degrees
    assert (target_course < 2 * np.pi) and (own_course < 2 * np.pi) 
    rel_speed_x = target_speed * np.sin(target_course) - own_speed * np.sin(own_course)
    rel_speed_y = target_speed * np.cos(target_course) - own_speed * np.cos(own_course)
    rel_speed_mag = np.linalg.norm([rel_speed_x, rel_speed_y])
    return rel_speed_x, rel_speed_y, rel_speed_mag