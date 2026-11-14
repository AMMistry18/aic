"""Independent wrist sampling selected causally against the policy ROS clock."""
from collections import deque
from copy import copy
import threading


class WristSensorCache:
    def __init__(self,node,topic='/fts_broadcaster/wrench',*,subscribe=True):
        self.node=node;self.topic=topic;self.latest=None;self.received=0;self.unique=0
        self.first_stamp=None;self.last_stamp=None;self.merges=0;self.max_age=0.
        self._samples=deque(maxlen=512);self._lock=threading.Lock()
        self.future_bypasses=0;self.max_latest_lead=0.
        if subscribe:
            from geometry_msgs.msg import WrenchStamped
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.callback_groups import ReentrantCallbackGroup
            self.callback_group=ReentrantCallbackGroup()
            self.subscription=node.create_subscription(WrenchStamped,topic,self.accept,qos_profile_sensor_data,
                                                       callback_group=self.callback_group)

    @staticmethod
    def stamp(message):
        return message.header.stamp.sec+message.header.stamp.nanosec*1e-9

    def accept(self,message):
        stamp=self.stamp(message)
        with self._lock:
            self.received+=1
            if self.last_stamp is not None and stamp<=self.last_stamp:return
            self.unique+=1;self.last_stamp=stamp
            if self.first_stamp is None:self.first_stamp=stamp
            self.latest=message;self._samples.append((stamp,message))

    def merge(self,observation):
        if observation is None:return None
        # /clock and the force stream travel independently. A newer force
        # callback can precede its /clock callback; retain history and select
        # the newest already-observed sample at or before the clock snapshot.
        # Do not consume future data, relabel it, widen age limits or wait
        # inside the motion command path. Camera exposures remain untouched.
        with self._lock:
            if not self._samples:raise RuntimeError('No independent wrist sensor sample')
            now=self.node.get_clock().now().nanoseconds*1e-9
            selected=next(((t,m) for t,m in reversed(self._samples) if t<=now),None)
            if selected is None:raise RuntimeError('No causal independent wrist sample at policy clock')
            stamp,message=selected;age=now-stamp
            if age>.1:raise RuntimeError(f'Independent wrist sample has invalid age {age:.4f}s')
            if message.header.frame_id.lstrip('/')!='ati/tool_link':
                raise RuntimeError(f'Unexpected independent wrist frame: {message.header.frame_id}')
            lead=self._samples[-1][0]-now
            if lead>0:self.future_bypasses+=1;self.max_latest_lead=max(self.max_latest_lead,lead)
            self.merges+=1;self.max_age=max(self.max_age,age)
        result=copy(observation);result.wrist_wrench=message
        return result

    def diagnostics(self):
        with self._lock:
            span=0. if self.first_stamp is None else self.last_stamp-self.first_stamp
            return {'topic':self.topic,'received':self.received,'unique_received':self.unique,
                'sensor_span_sim_s':span,'mean_received_hz':None if span<=0 else (self.unique-1)/span,
                'policy_merges':self.merges,'max_policy_age_sim_s':self.max_age,
                'future_latest_bypasses':self.future_bypasses,'max_latest_ahead_of_clock_s':self.max_latest_lead,
                'selection':'newest buffered sample at or before policy ROS clock',
                'buffer_capacity':512,'maximum_age_s':.1,'simulator_object_inputs':False}
