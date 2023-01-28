from os import close
from posix import times_result
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
from threading import Thread
import time  # mavlink mavproxy
import cv2
import numpy as np
import imutils
import math

home_pos_lat = 0
home_pos_alt = 0
home_pos_lon = 0
centerX = 0
centerY = 0

red_field_location = None

str_mode = ""

vehicle = None

close_flag = False
sleep_flag = False
first_mission_flag = True

# kırmızı icin set edilecek.
color_range_min = np.array([100, 135, 40])  # 166 152 89
color_range_max = np.array([115, 255, 255])  # 186 182 289


def Servo(pvm, slp):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  # komut
        0,  # confirmation
        1,  # servo port sayısı
        pvm,  # dönüş hızı ve yönü (1000 saat yönünde, 1500 durma, 2000 saat yönünün tersi)
        0, 0, 0, 0, 0)  # kullanılmıyor parametreler

    vehicle.send_mavlink(msg)
    time.sleep(slp)
    return


def connect_PX():
    print('Connecting...')
    global home_pos_alt, home_pos_lat, home_pos_lon, vehicle, str_mode
    vehicle = connect("127.0.0.1:14550", wait_ready = True)
    home_pos_lat = vehicle.location.global_relative_frame.lat
    home_pos_lon = vehicle.location.global_relative_frame.lon
    home_pos_alt = 3
    str_mode = 'GROUND'
    print("initializing...")


def arm_and_takeoff(tgt_altitude):
    global vehicle

    while not vehicle.is_armable:
        print("waiting to be armable")
        time.sleep(1)
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed: time.sleep(1)
    print("Takeoff")
    time.sleep(2)
    vehicle.simple_takeoff(tgt_altitude)
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(">> Altitude = %.1f m" % altitude)
        if altitude >= tgt_altitude - 1:
            print("Altitude reached")
            break
        time.sleep(1)


def set_velocity_body(vx, vy, vz):
    global vehicle
    """ Remember: vz is positive downward!!! """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
        0b0000111111000111,  # -- BITMASK -> Consider only the velocities
        0, 0, 0,  # -- POSITION
        vx, vy, vz,  # -- VELOCITY
        0, 0, 0,  # -- ACCELERATIONS
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def set_velocity_body_with_time(vx, vy, vz, duration):
    global vehicle
    """ Remember: vz is positive downward!!! """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
        0b0000111111000111,  # -- BITMASK -> Consider only the velocities
        0, 0, 0,  # -- POSITION
        vx, vy, vz,  # -- VELOCITY
        0, 0, 0,  # -- ACCELERATIONS
        0, 0)

    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


def clear_mission():
    global vehicle

    cmds = vehicle.commands
    vehicle.commands.clear()
    vehicle.flush()

    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()


def download_mission():
    global vehicle

    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()


def get_current_mission():
    global vehicle

    print("Downloading mission")
    download_mission()
    missionList = []
    n_WP = 0

    for wp in vehicle.commands:
        missionList.append(wp)
        n_WP += 1

    return n_WP, missionList


def add_last_waypoint_to_mission(wp_Last_Latitude, wp_Last_Longitude, wp_Last_Altitude):
    global vehicle

    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()

    missionList = []
    for cmd in cmds:
        missionList.append(cmd)

    wpLastObject = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0,
                           wp_Last_Latitude, wp_Last_Longitude, wp_Last_Altitude)

    missionList.append(wpLastObject)
    cmds.clear()
    for cmd in missionList:
        cmds.add(cmd)

    cmds.upload()
    return cmds.count


def ChangeMode(mode):
    global vehicle

    while vehicle.mode != VehicleMode(mode):
        vehicle.mode = VehicleMode(mode)
        time.sleep(0.1)

    return True


def Image_Processing_Cam():
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)

    global close_flag, color_range_max, color_range_min, sleep_flag, centerX, centerY, str_mode, vehicle, first_mission_flag

    while not close_flag:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range_min, color_range_max)

        # performansda çok düşüş yaşanırsa eğer bu fonksiyon iptal edilebilir.
        # başka bir alternatif olarak bounding rect alınarak yapılabilir.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

        """ kitleme ekranının büyüklüğü ve konumu ayarlanabilir. """
        frame = cv2.rectangle(frame, (300, 220), (340, 260), (255, 255, 255), 3)  # commentle gerek yok

        cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # görüntü işleme çalışıyor ama drone auto modda. Çünkü alan tespit edilmedi.
        # sleep_flag = False

        for c in cnts:
            area = cv2.contourArea(c)
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            M = cv2.moments(approx)

            # alan değişiklik gösterebilir. görev sayısı ikinci turun başladığı wp'den 2 önce falan olabilir.
            if area > 150 and vehicle.commands.next > 2 and first_mission_flag == True:  # -- değeri kontrol et gerçek sayısal değeri ver
                sleep_flag = True
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

                # global değişkenler değişiyor diğer thread burdan gelen verilere göre drone'u yönlendirecek.
                if M["m00"] != 0:
                    centerX = int(M["m10"] / M["m00"])
                    centerY = int(M["m01"] / M["m00"])
                else:
                    centerX = 0
                    centerY = 0

                # bu iki satıra yarışma esnasında hiç gerek yok ekranda orta noktayı göstermese de olur.
                cv2.circle(frame, (centerX, centerY), 7, (255, 255, 255), -1)  # commentle gerek yok
                cv2.putText(frame, "Center", (centerX - 20, centerY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)  # commentle gerek yok

            elif area > 150 and vehicle.commands.next > 6 and first_mission_flag == False:
                sleep_flag = True
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

                # global değişkenler değişiyor diğer thread burdan gelen verilere göre drone'u yönlendirecek.
                if M["m00"] != 0:
                    centerX = int(M["m10"] / M["m00"])
                    centerY = int(M["m01"] / M["m00"])
                else:
                    centerX = 0
                    centerY = 0

                # bu iki satıra hiç gerek yok ekranda orta noktayı göstermese de olur.
                cv2.circle(frame, (centerX, centerY), 7, (255, 255, 255), -1)  # commentle gerek yok
                cv2.putText(frame, "Center", (centerX - 20, centerY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)  # commentle gerek yok

        cv2.imshow("frame", frame)
        #cv2.imshow("mask", mask)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def UAV_Mission():
    global str_mode, sleep_flag, vehicle, close_flag, first_mission_flag, red_field_location

    while True:
        if str_mode == 'GROUND':
            n_WP, missionList = get_current_mission()
            time.sleep(2)
            if n_WP > 0:
                print("A valid mission has been uploaded: takeoff!")
                str_mode = 'TAKEOFF'

        elif str_mode == 'TAKEOFF':
            add_last_waypoint_to_mission(vehicle.location.global_relative_frame.lat,
                                         vehicle.location.global_relative_frame.lon,
                                         vehicle.location.global_relative_frame.alt)
            print("Home waypoint added to the mission")
            time.sleep(1)
            arm_and_takeoff(10)

            # -- Change the UAV mode to AUTO
            print("Changing to AUTO")
            ChangeMode("AUTO")

            str_mode = 'MISSION'
            print("Switch mode to MISSION")

        elif str_mode == 'MISSION':
            if sleep_flag:
                
                if first_mission_flag:
                    """
                    git koordinat al.
                    """
                    ChangeMode("GUIDED")

                    if centerX < 300 and centerY < 220:  # sol ust
                        set_velocity_body(3, -3, 0)
                    elif centerX < 300 and centerY > 260:  # sol alt
                        set_velocity_body(-3, -3, 0)
                    elif centerX > 340 and centerY < 220:  # sag ust
                        set_velocity_body(3, 3, 0)
                    elif centerX > 340 and centerY > 260:  # sag alt
                        set_velocity_body(-3, 3, 0)
                    elif centerX < 300:  # sol
                        set_velocity_body(0, -3, 0)
                    elif centerX > 340:  # sag
                        set_velocity_body(0, 3, 0)
                    elif centerY < 220:  # ust
                        set_velocity_body(3, 0, 0)
                    elif centerY > 260:  # alt
                        set_velocity_body(-3, 0, 0)
                    else:
                        time.sleep(2)
                        set_velocity_body_with_time(0, 0, 0.5, 6)  # vz * duration = deltaAltitude
                        red_field_location = LocationGlobalRelative(vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon, 6)
                        first_mission_flag = False
                        sleep_flag = False
                        ChangeMode("AUTO")

                else:
                    """
                    git suyu bırak
                    """
                    ChangeMode("GUIDED")

                    if centerX < 300 and centerY < 220:  # sol ust
                        set_velocity_body(5,-5, 0)
                    elif centerX < 300 and centerY > 260:  # sol alt
                        set_velocity_body(-5,-5, 0)
                    elif centerX > 340 and centerY < 220:  # sag ust
                        set_velocity_body(5, 5, 0)
                    elif centerX > 340 and centerY > 260:  # sag alt
                        set_velocity_body(-5, 5, 0)
                    elif centerX < 300:  # sol
                        set_velocity_body(0, -5, 0)
                    elif centerX > 340:  # sag
                        set_velocity_body(0, 5, 0)
                    elif centerY < 220:  # ust
                        set_velocity_body(5, 0, 0)
                    elif centerY > 260:  # alt
                        set_velocity_body(-5, 0, 0)
                    else:
                        time.sleep(2)  # degisebilir
                        set_velocity_body_with_time(0, 0, 0.5, 6)  # vz * duration = deltaAltitude
                        #Servo(2000, 10)  # hangi yöne kaç saniye düzenlenecek.  # bosaltmak için daha fazla sarmak gerekebilir. fonksiyon düzenlenebilir.
                        #Servo(1500, 1)
                        print("SERVO YERİ")
                        close_flag = True  # auto moda geçiş işlemleri imageProc içerisinde yapıldı.
                        sleep_flag = False
                        ChangeMode("AUTO")
            else:
                # -- Here we just monitor the mission status. Once the mission is completed we go back
                # -- vehicle.commands.cout is the total number of waypoints
                # -- vehicle.commands.next is the waypoint the vehicle is going to
                # -- once next == cout, we just go home
                print("Current WP: %d of %d " % (vehicle.commands.next, vehicle.commands.count))
                if vehicle.commands.next == vehicle.commands.count:
                    print("Final waypoint reached: go back home")
                    # -- First we clear the flight mission
                    clear_mission()
                    print("Mission deleted")
                    # -- We go back home
                    ChangeMode("LAND")
                    str_mode = "LAND"

        elif str_mode == "LAND":
            if vehicle.location.global_relative_frame.alt < 1:  # vehicle sonradan eklendi
                print("Switch to GROUND mode, waiting for new missions")  # 2. görev için kullanılacak
                str_mode = "FINISH"  # ground olunca 2. görev

        elif str_mode == "FINISH":
            print(red_field_location)
            vehicle.armed = False
            vehicle.close()

        time.sleep(0.01)


"""
thread olusturulacak UAV_Mission ve Image_Processing_Cam ayrı ayrı calıstırılacak.
"""
connect_PX()
t1 = Thread(target = Image_Processing_Cam)
t2 = Thread(target = UAV_Mission)
t1.start()
t2.start()
