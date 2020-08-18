#!/usr/bin/env python
import rospy
# from std_msgs.msg import simulator_control
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from transformations import quaternion_from_euler, euler_from_quaternion
from DubinsCar import DubinsCar as model
running = False
init_state = None
state = None
step = 0

def callback(data):
    global running, init_state, state, step
    init_state = data.data['init_state']
    state = init_state
    step = 0
    # running = True
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

state = np.array([0,0,0])
step = 0

if __name__ == '__main__':
    rospy.init_node('simulator', anonymous=True)
    rate = rospy.Rate(1000) # we only not need real time for visualization
    # pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    rospy.Subscriber("simulator_control", simulator_control, callback)

    while not rospy.is_shutdown():
        if step > model.k:
            continue
        unsafe = model.is_unsafe(state.tolist())
        if unsafe:
            continue
        state = model.transition(state)
        step += 1
        # rospy.loginfo(hello_str)
        state_msg = ModelState()
        state_msg.model_name = 'car'
        state_msg.pose.position.x = state[0]
        state_msg.pose.position.y = state[1]
        state_msg.pose.position.z = 0.
        quaternion = quaternion_from_euler(0,0,state[2], 'szxy')
        state_msg.pose.orientation.x = quaternion[1]
        state_msg.pose.orientation.y = quaternion[2]
        state_msg.pose.orientation.z = quaternion[3]
        state_msg.pose.orientation.w = quaternion[0]
        state_msg.reference_frame = 'world'
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        try:
            # pub.publish(hello_str)
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
