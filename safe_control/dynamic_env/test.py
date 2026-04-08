import rospy
from gazebo_msgs.msg import ModelStates

def callback(msg):
    try:
        i = msg.name.index("jackal")
        pose = msg.pose[i]

        print("X:", pose.position.x)
        print("Y:", pose.position.y)
        print("Z:", pose.position.z)

    except ValueError:
        print("Jackal not found")

rospy.init_node("jackal_pose_reader")
rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
rospy.spin()