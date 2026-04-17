using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using System;

public class G1mimic1Agent : Agent
{
    [Header("Mode")]
    public bool train = false;
    public bool replay = false;

    // --- 新增：奖励函数的权重与参数 ---
    [Header("Standing Reward")]
    public bool enable_standing_reward = true;  
    public float standing_reward_weight = 0.3f; 
    public int standing_start_frame = -1;       

    private float w_pose = 0.45f;
    private float w_vel = 0.15f;
    private float w_root_rot = 0.12f;
    private float w_root_pos = 0.08f;
    private float w_upright = 0.10f;
    private float w_action_smooth = 0.05f;
    private float w_alive = 0.05f;
    private float terminate_height = 0.45f;

    // --- 新增：平滑控制与站立参考数据 ---
    private float[] action_filtered = new float[29];
    private float[] action_prev = new float[29];
    private float[] nextDof = new float[29];
    private float[] _standDof = new float[29];
    private bool _hasStandDof = false;

    float[] uff = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] u = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] utotal = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    int[] trainid=new int[13]{5,5,13,13,   6,14,15,   5,5,13,13,   16,17};

    private List<string> actionFolders = new List<string>();
    public int motion_id;
    public string motion_name;

    private string dofFilePath;
    private string rotFilePath;
    private string posFilePath;

    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();

    public int currentFrame;

    float[] currentData = new float[36];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[29];

    Transform body;

    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[29];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;
    public int frame0 = 500;
    public bool rand = true;
    int endT = 0;
    int idx = 0;

    private bool _isClone = false;

    void Start()
    {
        Time.fixedDeltaTime = 0.01f;
        if (train && !_isClone)
        {
            for (int i = 1; i < 128; i++)
            {
                GameObject clone = Instantiate(gameObject);
                clone.transform.position = transform.position + new Vector3(i * 2f, 0, 0);
                clone.name = $"{name}_Clone_{i}";
                clone.GetComponent<G1mimic1Agent>()._isClone = true;
            }
        }

        arts = this.GetComponentsInChildren<ArticulationBody>();
        int ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if(arts[k].jointType.ToString() == "RevoluteJoint")
            {
                jh[ActionNum] = arts[k];
                ActionNum++;
            }
        }

        body = arts[0].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        pos0 = body.position;
        rot0 = body.rotation;

        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        List<string> csvFileNames = GetCsvFileNames(streamingAssetsPath);
        refData = LoadDataFromFile(csvFileNames[motion_id]);
        float[] refT = new float[refData.Count];
        for(int i=0;i<refT.Length;i++)refT[i]=i/30f;
        float[] newT = new float[(int)(refData.Count*100f/30f)-5];
        for(int i=0;i<newT.Length;i++)newT[i]=i/100f;
        itpData = Interpolate(refT, refData, newT);
        
        motion_name = csvFileNames[motion_id].Replace("./Assets/Imitation/G1/dataset\\", "").Replace(".csv", "");

        // --- 新增：获取站立姿态参考（使用第一帧的关节角度） ---
        if (itpData != null && itpData.Count > 0)
        {
            float[] firstFrame = itpData[0];
            Array.Copy(firstFrame, 7, _standDof, 0, 29);
            _hasStandDof = true;
        }
    }

    List<string> GetCsvFileNames(string directoryPath)
    {
        List<string> csvFiles = new List<string>();

        try
        {
            if (Directory.Exists(directoryPath))
            {
                string[] allFiles = Directory.GetFiles(directoryPath);
                foreach (string file in allFiles)
                {
                    if (Path.GetExtension(file).ToLower() == ".csv")
                    {
                        string fileName = Path.GetFileName(file);
                        csvFiles.Add(Path.Combine(directoryPath, fileName));
                    }
                }
            }
            else
            {
                UnityEngine.Debug.LogError("Directory does not exist: " + directoryPath);
            }
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError("Error accessing directory: " + e.Message);
        }
        return csvFiles;
    }

    List<float[]> LoadDataFromFile(string filePath)
    {
        List<float[]> dataList = new List<float[]>();
        try
        {
            string[] lines = File.ReadAllLines(filePath);
            foreach (string line in lines)
            {
                string[] values = line.Split(',');
                List<float> frameData = new List<float>();
                foreach (string value in values)
                {
                    if (float.TryParse(value.Trim(), out float parsedValue))frameData.Add(parsedValue);
                }
                dataList.Add(frameData.ToArray());
            }
        }
        catch (System.Exception e)
        {
            print("Error loading data from file " + filePath + ": " + e.Message);
        }
        return dataList;
    }

    public override void OnEpisodeBegin()
    {
        for (int i = 0; i < 29; i++) u[i] = 0;
        for (int i = 0; i < 29; i++) uff[i] = 0;

        // --- 新增：重置平滑控制数组 ---
        Array.Clear(action_filtered, 0, 29);
        Array.Clear(action_prev, 0, 29);

        if(endT>450)idx=(idx+1)%5;
        if(rand)frame0 = 1000 + Random.Range(0,5)*100;
        
        currentFrame = frame0 % itpData.Count;

        tt=0;
        currentData = itpData[currentFrame];
        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, 29);

        Vector3 newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        Quaternion newRotation = new Quaternion(
            -currentRot[1],
            currentRot[2],
            currentRot[0],
            -currentRot[3]
        );

        arts[0].TeleportRoot(newPosition, newRotation);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;
        float[] Dof = new float[35]{0,0,0,0,0,0,   currentDof[12], currentDof[6], currentDof[0], currentDof[13], currentDof[7], currentDof[1], currentDof[14], currentDof[8], currentDof[2], currentDof[15], currentDof[22], currentDof[9], currentDof[3], currentDof[16], currentDof[23], currentDof[10], currentDof[4], currentDof[17], currentDof[24], currentDof[11], currentDof[5], currentDof[18], currentDof[25], currentDof[19], currentDof[26], currentDof[20], currentDof[27], currentDof[21], currentDof[28]};
        List<float> jointPositions = new List<float>();
        for (int i = 0; i < 29+6; i++)jointPositions.Add(Dof[i]);
        arts[0].SetJointPositions(jointPositions);
    }

    List<float[]> Interpolate(float[] t, List<float[]> posList, float[] targetT)
    {
        if (t.Length != posList.Count)
        {
            UnityEngine.Debug.LogError("t and posList must have the same length");
            return null;
        }
        int dimension = posList[0].Length;
        foreach (float[] arr in posList)
        {
            if (arr.Length != dimension)
            {
                UnityEngine.Debug.LogError("All arrays in posList must have the same length");
                return null;
            }
        }
        List<float[]> result = new List<float[]>();
        for (int i = 0; i < targetT.Length; i++)
        {
            float tValue = targetT[i];
            if (tValue < t[0] || tValue > t[t.Length - 1])
            {
                UnityEngine.Debug.LogError("tValue is out of range");
                return null;
            }
            int index = 0;
            while (index < t.Length - 1 && t[index + 1] < tValue)
            {
                index++;
            }
            float ratio = (tValue - t[index]) / (t[index + 1] - t[index]);
            float[] interpolatedPos = new float[dimension];
            for (int j = 0; j < dimension; j++)
            {
                interpolatedPos[j] = Mathf.Lerp(posList[index][j], posList[index + 1][j], ratio);
            }
            result.Add(interpolatedPos);
        }
         return result;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * 3.14f / 180f );//rad
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * 3.14f / 180f );//rad
        sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        
        for (int i = 0; i < 29; i++)
        {
            sensor.AddObservation(jh[i].jointPosition[0]);
            sensor.AddObservation(jh[i].jointVelocity[0]);
        }
        
        Vector3 epos = body.position - newPosition;
        Vector3 newEuler = newRotation.eulerAngles;
        sensor.AddObservation(epos);
        sensor.AddObservation(newEuler.x);
        sensor.AddObservation(newEuler.z);
    }

    float EulerTrans(float angle)
    {
        angle = angle % 360f;
        if (angle > 180f)angle -= 360f;
        else if (angle < -180f)angle += 360f;
        return angle;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (itpData.Count > 0)
        {
            currentData = itpData[currentFrame % itpData.Count];
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, 29);
            for (int i = 0; i < 29; i++)uff[i] = currentDof[i]* 180f / 3.14f;

            newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
            newRotation = new Quaternion(
                -currentRot[1],
                currentRot[2],
                currentRot[0],
                -currentRot[3]
            );

            if(replay)
            {
                arts[0].TeleportRoot(newPosition, newRotation);
                arts[0].velocity = Vector3.zero;
                arts[0].angularVelocity = Vector3.zero;
                float[] Dof = new float[35]{0,0,0,0,0,0,   currentDof[12], currentDof[6], currentDof[0], currentDof[13], currentDof[7], currentDof[1], currentDof[14], currentDof[8], currentDof[2], currentDof[15], currentDof[22], currentDof[9], currentDof[3], currentDof[16], currentDof[23], currentDof[10], currentDof[4], currentDof[17], currentDof[24], currentDof[11], currentDof[5], currentDof[18], currentDof[25], currentDof[19], currentDof[26], currentDof[20], currentDof[27], currentDof[21], currentDof[28]};
                List<float> jointPositions = new List<float>();
                for (int i = 0; i < 29+6; i++)jointPositions.Add(Dof[i]);
                arts[0].SetJointPositions(jointPositions);
            }
        }

        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        float kb = 40;
        
        for (int i = 0; i < 29; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            if(i>=15)kb=0;
            if(replay)kb = 0;
            utotal[i] = kb * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);

            // --- 新增：记录滤波后的动作，用于平滑度奖励计算 ---
            action_filtered[i] = u[i];
        }
    }

    void FixedUpdate()
    {
        tt++;

        // ===== 参考数据 =====
        currentData = itpData[currentFrame % itpData.Count];
        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, 29);

        newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        newRotation = new Quaternion(
            -currentRot[1],
            currentRot[2],
            currentRot[0],
            -currentRot[3]
        );

        // ============================================
        // ===== 替换后的奖励函数与终止条件开始 =====
        // ============================================
        if (tt > 1)
        {
            // 1. Pose Reward (关节位置误差)
            float pose_reward = 0f;
            for (int i = 0; i < 29; i++)
            {
                float jointError = Mathf.Abs(jh[i].jointPosition[0] - currentDof[i]);
                pose_reward += Mathf.Exp(-5f * jointError * jointError);
            }
            pose_reward /= 29f;

            // 2. Velocity Reward (关节速度误差)
            float vel_reward = 1f;
            if (itpData.Count > 0)
            {
                vel_reward = 0f;
                // 安全获取下一帧的数据
                Array.Copy(itpData[(currentFrame + 1) % itpData.Count], 7, nextDof, 0, 29);
                for (int i = 0; i < 29; i++)
                {
                    float targetVel = (nextDof[i] - currentDof[i]) * 100f;
                    float velError = Mathf.Abs(jh[i].jointVelocity[0] - targetVel);
                    vel_reward += Mathf.Exp(-0.02f * velError * velError);
                }
                vel_reward /= 29f;
            }

            // 3. Root Rotation Reward (根节点旋转误差)
            Vector3 bodyEuler = body.eulerAngles;
            Vector3 refEuler = newRotation.eulerAngles;
            float rotError = (Mathf.Abs(EulerTrans(bodyEuler.x) - EulerTrans(refEuler.x)) + 
                              Mathf.Abs(EulerTrans(bodyEuler.z) - EulerTrans(refEuler.z))) * Mathf.PI / 360f;
            float root_rot_reward = Mathf.Exp(-2f * rotError * rotError);

            // 4. Root Position Reward (根节点位置误差)
            float posError = (body.position - newPosition).magnitude;
            float root_pos_reward = Mathf.Exp(-2f * posError * posError);

            // 5. Upright Reward (直立奖励)
            float tiltAngle = (Mathf.Abs(EulerTrans(bodyEuler.x)) + Mathf.Abs(EulerTrans(bodyEuler.z))) * Mathf.PI / 360f;
            float upright_reward = Mathf.Exp(-0.5f * tiltAngle * tiltAngle);

            // 6. Action Smoothness Reward (动作平滑度惩罚)
            float action_smooth_reward = 0f;
            for (int i = 0; i < 29; i++)
            {
                action_smooth_reward += 1f - Mathf.Clamp01(Mathf.Abs(action_filtered[i] - action_prev[i]));
            }
            action_smooth_reward /= 29f;

            // --- 组合基础奖励 ---
            float reward = w_pose * pose_reward + 
                           w_vel * vel_reward + 
                           w_root_rot * root_rot_reward +
                           w_root_pos * root_pos_reward + 
                           w_upright * upright_reward + 
                           w_action_smooth * action_smooth_reward + 
                           w_alive * 1f;

            // --- 附加：站立奖励 ---
            if (enable_standing_reward && _hasStandDof)
            {
                bool shouldGiveStandingReward = false;
                
                if (standing_start_frame >= 0)
                {
                    shouldGiveStandingReward = currentFrame >= standing_start_frame;
                }
                else
                {
                    // 在循环动作中，这通常设定为你希望开始鼓励站立的帧
                    // 你的原代码中是取余数循环的，这里假设满足某种条件（例如在动作末尾）
                    // 为了适配你的无尽循环逻辑，默认开启或依靠 standing_start_frame 控制
                    shouldGiveStandingReward = true; 
                }
                
                if (shouldGiveStandingReward)
                {
                    float standPoseReward = 0f;
                    for (int i = 0; i < 29; i++)
                    {
                        float jointError = Mathf.Abs(jh[i].jointPosition[0] - _standDof[i]);
                        standPoseReward += Mathf.Exp(-5f * jointError * jointError);
                    }
                    standPoseReward /= 29f;
                    
                    float standUprightReward = Mathf.Exp(-2f * tiltAngle * tiltAngle);
                    float standStillReward = Mathf.Exp(-2f * art0.velocity.magnitude);
                    
                    float standingReward = (standPoseReward * 0.5f + standUprightReward * 0.3f + standStillReward * 0.2f);
                    reward += standingReward * standing_reward_weight;
                }
            }

            // --- 终止条件 ---
            if (train && !replay)
            {
                if (body.position.y < terminate_height || 
                    Mathf.Abs(EulerTrans(bodyEuler.x)) > 80f || 
                    Mathf.Abs(EulerTrans(bodyEuler.z)) > 80f)
                {
                    EndEpisode();
                }
            }

            AddReward(reward);
        }

        // 记录本帧动作，用于下一帧的平滑度计算
        Array.Copy(action_filtered, action_prev, 29);

        // ===== 帧推进 =====
        currentFrame++;
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f; //2000f;
        drive.damping = 8f;     //200f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {

    }
}