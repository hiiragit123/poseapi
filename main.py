import mediapipe as mp
import cv2 
import math

class pose_app:
    def __init__(self): 
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5)
    
    def get_image(self,ans,input):
        ans_image = cv2.imdecode(ans, cv2.IMREAD_COLOR)
        input_image = cv2.imdecode(input, cv2.IMREAD_COLOR)
        ans_image = cv2.cvtColor(ans_image,cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)

        return ans_image,input_image
    
    def get_pose(self,ans,input):
        result_ans = self.pose.process(ans)
        result_input = self.pose.process(input)

        return result_input.pose_world_landmarks,result_ans.pose_world_landmarks

    def get_dist(self,ans,input):
        ans_mark = ans
        input_mark = input
        dist = []
        for i in range(33):
            ax = ans.landmark[i].x
            ay = ans.landmark[i].y
            az = ans.landmark[i].z
            ix = input.landmark[i].x
            iy = input.landmark[i].y
            iz = input.landmark[i].z
            distance = math.sqrt(pow((ax-ix),2)+pow((ay-iy),2)+pow((az-iz),2))
            dist.append(distance)
        print(dist)
        return dist
    
    def get_similarity(self,dist):
        dist = dist
        similarity = 0
        for d in dist:
            if d < 0.15:
                similarity+=3
            elif d >= 0.15 and d < 0.5:
                similarity+=2
            else:
                continue
        
        return similarity
    
    def judge(self,similarity):
        if similarity > 70:
            return 2
        elif similarity <= 70 and similarity > 50:
            return 1
        else:
            return 0