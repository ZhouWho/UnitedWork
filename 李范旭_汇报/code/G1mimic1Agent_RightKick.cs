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

    [Header("Motion")]
    public int motion_id = 0;
    public string motion_name;
    public bool rand_start = true;
    public int frame_start = 6000;   // ← 右踢腿默认值：CSV 1800帧 × (100/30)
    public int frame_end = 6667;   // ← 右踢腿默认值：CSV 2000帧 × (100/30)
    public bool loop_playback = false; // ← 踢腿是单次动作，默认关闭往返播放
    public int frame0 = 0;

    [Header("Standing Reward")]
    public bool enable_standing_reward = true;
    public float standing_reward_weight = 0.3f;
    public int standing_start_frame = -1;

    private float actionSmooth = 0.90f;
    private float stiffness = 180f;
    private float damping = 8f;

    // [修改3] 调整权重：更适合踢腿动作
    private float w_pose = 0.50f;   // 原 0.45 → 关节姿态最重要
    private float w_vel = 0.15f;   // 不变
    private float w_root_rot = 0.10f;   // 原 0.12 → 踢腿时躯干轻微旋转属正常
    private float w_root_pos = 0.08f;   // 不变
    private float w_upright = 0.05f;   // 原 0.10 → 踢腿时身体会侧倾，放宽
    private float w_action_smooth = 0.07f;   // 原 0.05 → 防腿部关节抖动
    private float w_alive = 0.05f;   // 不变
    private float terminate_height = 0.45f;

    [Header("Disturbance")]
    public bool add_disturbance = true;
    private float min_disturbance_interval = 2f;
    private float max_disturbance_interval = 5f;
    private float min_disturbance_force = 5f;
    private float max_disturbance_force = 20f;
    private float disturbance_torque = 5f;

    // 站立姿态参考（[修改2] 改为从 itpData[0] 获取）
    private float[] _standDof = new float[29];
    private bool _hasStandDof = false;

    private float[] action_filtered = new float[29];
    private float[] action_prev = new float[29];
    float[] uff = new float[29];
    float[] u = new float[29];
    float[] utotal = new float[29];

    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();

    public int currentFrame;

    float[] currentData = new float[36];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[29];
    float[] nextDof = new float[29];

    Transform body;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[29];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;

    private int _currentLoopFrame = 0;
    private float _disturbanceTimer = 0f;
    private float _nextDisturbanceInterval = 3f;
    private bool _isClone = false;

    // =========================================================
    //  Start
    // =========================================================
    void Start()
    {
        Time.fixedDeltaTime = 0.01f;

        if (train && !_isClone)
        {
            for (int i = 1; i < 34; i++)
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
            if (arts[k].jointType.ToString() == "RevoluteJoint")
            {
                jh[ActionNum] = arts[k];
                ActionNum++;
            }
        }

        body = arts[0].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        List<string> csvFileNames = GetCsvFileNames(streamingAssetsPath);

        if (csvFileNames.Count > 0)
        {
            motion_id = Mathf.Clamp(motion_id, 0, csvFileNames.Count - 1);
            refData = LoadDataFromFile(csvFileNames[motion_id]);

            float[] refT = new float[refData.Count];
            for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;
            float[] newT = new float[(int)(refData.Count * 100f / 30f) - 5];
            for (int i = 0; i < newT.Length; i++) newT[i] = i / 100f;
            itpData = Interpolate(refT, refData, newT);

            motion_name = Path.GetFileNameWithoutExtension(csvFileNames[motion_id]);
            Debug.Log($"Loaded {motion_name}, total itpData frames: {itpData.Count}");

            frame_end = Mathf.Min(frame_end, itpData.Count - 1);
            frame_start = Mathf.Min(frame_start, frame_end);

            // [修改2] 站立参考取 itpData[0]（CSV 第0帧 = 自然站立姿态）
            // 原代码取 frame_start，那是踢腿起始帧，不是站立姿态
            if (itpData.Count > 0)
            {
                float[] standFrame = itpData[0];
                Array.Copy(standFrame, 7, _standDof, 0, 29);
                _hasStandDof = true;
                Debug.Log("Standing pose reference loaded from itpData[0] (neutral stand)");
            }
        }
    }

    // =========================================================
    //  文件工具
    // =========================================================
    List<string> GetCsvFileNames(string directoryPath)
    {
        List<string> csvFiles = new List<string>();
        try
        {
            if (Directory.Exists(directoryPath))
            {
                string[] allFiles = Directory.GetFiles(directoryPath);
                foreach (string file in allFiles)
                    if (Path.GetExtension(file).ToLower() == ".csv")
                        csvFiles.Add(Path.Combine(directoryPath, Path.GetFileName(file)));
            }
        }
        catch (Exception e) { Debug.LogError("Error: " + e.Message); }
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
                    if (float.TryParse(value.Trim(), out float parsedValue))
                        frameData.Add(parsedValue);
                dataList.Add(frameData.ToArray());
            }
        }
        catch (Exception e) { print("Error: " + e.Message); }
        return dataList;
    }

    // =========================================================
    //  OnEpisodeBegin
    // =========================================================
    public override void OnEpisodeBegin()
    {
        Array.Clear(uff, 0, 29);
        Array.Clear(u, 0, 29);
        Array.Clear(utotal, 0, 29);
        Array.Clear(action_filtered, 0, 29);
        Array.Clear(action_prev, 0, 29);

        _disturbanceTimer = 0f;
        _nextDisturbanceInterval = Random.Range(min_disturbance_interval, max_disturbance_interval);

        // 设置起始帧
        if (frame_end > frame_start)
        {
            if (loop_playback)
            {
                if (rand_start)
                {
                    _currentLoopFrame = Random.Range(0, (frame_end - frame_start) * 2);
                    UpdateFrameFromLoop();
                }
                else
                {
                    _currentLoopFrame = 0;
                    currentFrame = frame_start;
                }
            }
            else
            {
                currentFrame = rand_start
                    ? Random.Range(frame_start, frame_end)
                    : frame_start;
            }
        }
        else
        {
            currentFrame = frame0;
        }

        tt = 0;
        UpdateCurrentFrameData();

        newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        newRotation = new Quaternion(-currentRot[1], currentRot[2], currentRot[0], -currentRot[3]);

        arts[0].TeleportRoot(newPosition, newRotation);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;

        float[] Dof = new float[35]{ 0, 0, 0, 0, 0, 0,
            currentDof[12], currentDof[6],  currentDof[0],
            currentDof[13], currentDof[7],  currentDof[1],
            currentDof[14], currentDof[8],  currentDof[2],
            currentDof[15], currentDof[22], currentDof[9],
            currentDof[3],  currentDof[16], currentDof[23],
            currentDof[10], currentDof[4],  currentDof[17],
            currentDof[24], currentDof[11], currentDof[5],
            currentDof[18], currentDof[25], currentDof[19],
            currentDof[26], currentDof[20], currentDof[27],
            currentDof[21], currentDof[28] };

        List<float> jointPositions = new List<float>();
        for (int i = 0; i < 35; i++) jointPositions.Add(Dof[i]);
        arts[0].SetJointPositions(jointPositions);
    }

    // =========================================================
    //  UpdateFrameFromLoop（往返播放，不变）
    // =========================================================
    void UpdateFrameFromLoop()
    {
        int range = frame_end - frame_start;
        if (_currentLoopFrame < range)
            currentFrame = frame_start + _currentLoopFrame;
        else
            currentFrame = frame_end - (_currentLoopFrame - range);
    }

    // =========================================================
    //  Interpolate（不变）
    // =========================================================
    List<float[]> Interpolate(float[] t, List<float[]> posList, float[] targetT)
    {
        List<float[]> result = new List<float[]>();
        int dimension = posList[0].Length;
        for (int i = 0; i < targetT.Length; i++)
        {
            float tValue = targetT[i];
            int index = 0;
            while (index < t.Length - 1 && t[index + 1] < tValue) index++;
            float ratio = (tValue - t[index]) / (t[index + 1] - t[index]);
            float[] interpolatedPos = new float[dimension];
            for (int j = 0; j < dimension; j++)
                interpolatedPos[j] = Mathf.Lerp(posList[index][j], posList[index + 1][j], ratio);
            result.Add(interpolatedPos);
        }
        return result;
    }

    // =========================================================
    //  CollectObservations（不变）
    //  观测维度：2 + 3 + 29 + 29 + 29 + 29 + 3 + 2 = 126
    // =========================================================
    public override void CollectObservations(VectorSensor sensor)
    {
        // 躯干姿态
        sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * Mathf.PI / 180f);
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * Mathf.PI / 180f);
        sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));

        // 关节位置
        for (int i = 0; i < 29; i++)
            sensor.AddObservation(jh[i].jointPosition[0]);

        // 关节速度
        for (int i = 0; i < 29; i++)
            sensor.AddObservation(jh[i].jointVelocity[0]);

        // 当前帧参考关节角
        for (int i = 0; i < 29; i++)
            sensor.AddObservation(currentDof[i]);

        // 下一帧参考速度（给网络预判能力）
        if (currentFrame + 1 < itpData.Count)
        {
            Array.Copy(itpData[currentFrame + 1], 7, nextDof, 0, 29);
            for (int i = 0; i < 29; i++)
                sensor.AddObservation((nextDof[i] - currentDof[i]) * 100f);
        }
        else
        {
            for (int i = 0; i < 29; i++)
                sensor.AddObservation(0f);
        }

        // 根节点位置误差 + 参考朝向
        Vector3 epos = body.position - newPosition;
        Vector3 newEuler = newRotation.eulerAngles;
        sensor.AddObservation(epos);
        sensor.AddObservation(newEuler.x);
        sensor.AddObservation(newEuler.z);
    }

    float EulerTrans(float angle)
    {
        angle = angle % 360f;
        if (angle > 180f) angle -= 360f;
        else if (angle < -180f) angle += 360f;
        return angle;
    }

    // =========================================================
    //  OnActionReceived  ← [修改1] 修复腿部关节 kb 被清零的 bug
    // =========================================================
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (itpData.Count > 0)
        {
            UpdateCurrentFrameData();
            for (int i = 0; i < 29; i++) uff[i] = currentDof[i] * 180f / Mathf.PI;

            newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
            newRotation = new Quaternion(-currentRot[1], currentRot[2], currentRot[0], -currentRot[3]);

            if (replay)
            {
                arts[0].TeleportRoot(newPosition, newRotation);
                arts[0].velocity = Vector3.zero;
                arts[0].angularVelocity = Vector3.zero;

                float[] Dof = new float[35]{ 0, 0, 0, 0, 0, 0,
                    currentDof[12], currentDof[6],  currentDof[0],
                    currentDof[13], currentDof[7],  currentDof[1],
                    currentDof[14], currentDof[8],  currentDof[2],
                    currentDof[15], currentDof[22], currentDof[9],
                    currentDof[3],  currentDof[16], currentDof[23],
                    currentDof[10], currentDof[4],  currentDof[17],
                    currentDof[24], currentDof[11], currentDof[5],
                    currentDof[18], currentDof[25], currentDof[19],
                    currentDof[26], currentDof[20], currentDof[27],
                    currentDof[21], currentDof[28] };

                List<float> jointPositions = new List<float>();
                for (int i = 0; i < 35; i++) jointPositions.Add(Dof[i]);
                arts[0].SetJointPositions(jointPositions);
            }
        }

        var continuousActions = actionBuffers.ContinuousActions;

        // [修改1] 原代码：在循环内 "if (i >= 15 || replay) kb = 0;"
        //   → kb 一旦变0就不会恢复，导致关节 15-28（腿部）永远 kb=0
        //   → RL策略对所有腿部关节完全无效，根本无法学踢腿
        //
        // 修复：每个关节独立判断自己的 kb_i，互不影响
        for (int i = 0; i < 29; i++)
        {
            u[i] = u[i] * actionSmooth + (1f - actionSmooth) * continuousActions[i];

            // replay 模式：纯前馈，不叠加RL输出
            // 训练模式：所有关节（含腿部 i>=15）都叠加 RL 修正
            float kb_i = replay ? 0f : 40f;

            utotal[i] = kb_i * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
            action_filtered[i] = u[i];
        }
    }

    // =========================================================
    //  FixedUpdate：随机干扰 + 奖励计算
    //  奖励函数结构不变，只调整了权重（见文件头说明）
    // =========================================================
    void FixedUpdate()
    {
        tt++;

        // ── 随机外力干扰（抗打击训练）──────────────────────
        if (add_disturbance)
        {
            _disturbanceTimer += Time.fixedDeltaTime;
            if (_disturbanceTimer >= _nextDisturbanceInterval)
            {
                _disturbanceTimer = 0f;
                _nextDisturbanceInterval = Random.Range(min_disturbance_interval, max_disturbance_interval);

                float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
                Vector3 randomDir = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
                float force = Random.Range(min_disturbance_force, max_disturbance_force);
                art0.AddForce(randomDir * force, ForceMode.Impulse);

                if (disturbance_torque > 0)
                {
                    Vector3 randomTorque = Random.insideUnitSphere.normalized * disturbance_torque;
                    art0.AddTorque(randomTorque, ForceMode.Impulse);
                }
            }
        }

        if (tt > 1)
        {
            // ── 1. 关节姿态奖励 ───────────────────────────
            float pose_reward = 0f;
            for (int i = 0; i < 29; i++)
            {
                float jointError = Mathf.Abs(jh[i].jointPosition[0] - currentDof[i]);
                pose_reward += Mathf.Exp(-5f * jointError * jointError);
            }
            pose_reward /= 29f;

            // ── 2. 关节速度奖励 ───────────────────────────
            float vel_reward = 1f;
            if (currentFrame + 1 < itpData.Count)
            {
                vel_reward = 0f;
                Array.Copy(itpData[currentFrame + 1], 7, nextDof, 0, 29);
                for (int i = 0; i < 29; i++)
                {
                    float targetVel = (nextDof[i] - currentDof[i]) * 100f;
                    float velError = Mathf.Abs(jh[i].jointVelocity[0] - targetVel);
                    vel_reward += Mathf.Exp(-0.02f * velError * velError);
                }
                vel_reward /= 29f;
            }

            // ── 3. 根节点朝向奖励 ─────────────────────────
            Vector3 bodyEuler = body.eulerAngles;
            Vector3 refEuler = newRotation.eulerAngles;
            float rotError = (Mathf.Abs(EulerTrans(bodyEuler.x) - EulerTrans(refEuler.x)) +
                                 Mathf.Abs(EulerTrans(bodyEuler.z) - EulerTrans(refEuler.z)))
                                * Mathf.PI / 360f;
            float root_rot_reward = Mathf.Exp(-2f * rotError * rotError);

            // ── 4. 根节点位置奖励 ─────────────────────────
            float posError = (body.position - newPosition).magnitude;
            float root_pos_reward = Mathf.Exp(-2f * posError * posError);

            // ── 5. 直立奖励（[修改3] w_upright 降低，踢腿时允许侧倾）
            float tiltAngle = (Mathf.Abs(EulerTrans(bodyEuler.x)) +
                                  Mathf.Abs(EulerTrans(bodyEuler.z))) * Mathf.PI / 360f;
            float upright_reward = Mathf.Exp(-0.5f * tiltAngle * tiltAngle);

            // ── 6. 动作平滑奖励（[修改3] w_action_smooth 提高，防腿部抖动）
            float action_smooth_reward = 0f;
            for (int i = 0; i < 29; i++)
                action_smooth_reward += 1f - Mathf.Clamp01(Mathf.Abs(action_filtered[i] - action_prev[i]));
            action_smooth_reward /= 29f;

            // ── 加权总奖励 ────────────────────────────────
            float reward = w_pose * pose_reward
                         + w_vel * vel_reward
                         + w_root_rot * root_rot_reward
                         + w_root_pos * root_pos_reward
                         + w_upright * upright_reward
                         + w_action_smooth * action_smooth_reward
                         + w_alive * 1f;

            // ── 站立奖励（动作结束后）─────────────────────
            if (enable_standing_reward && _hasStandDof)
            {
                bool shouldGiveStandingReward = (standing_start_frame >= 0)
                    ? currentFrame >= standing_start_frame
                    : currentFrame >= frame_end;

                if (shouldGiveStandingReward)
                {
                    // 关节姿态接近站立姿态
                    float standPoseReward = 0f;
                    for (int i = 0; i < 29; i++)
                    {
                        float jointError = Mathf.Abs(jh[i].jointPosition[0] - _standDof[i]);
                        standPoseReward += Mathf.Exp(-5f * jointError * jointError);
                    }
                    standPoseReward /= 29f;

                    // 保持直立
                    float standUprightReward = Mathf.Exp(-2f * tiltAngle * tiltAngle);

                    // 减少移动（站在原地）
                    float standStillReward = Mathf.Exp(-2f * art0.velocity.magnitude);

                    float standingReward = standPoseReward * 0.5f
                                        + standUprightReward * 0.3f
                                        + standStillReward * 0.2f;

                    reward += standingReward * standing_reward_weight;
                }
            }

            // ── 终止条件 ──────────────────────────────────
            if (body.position.y < terminate_height
                || Mathf.Abs(EulerTrans(bodyEuler.x)) > 80f
                || Mathf.Abs(EulerTrans(bodyEuler.z)) > 80f)
            {
                if (train && !replay) EndEpisode();
            }

            AddReward(reward);
        }

        Array.Copy(action_filtered, action_prev, 29);

        // ── 帧更新逻辑 ────────────────────────────────────
        if (frame_end > frame_start)
        {
            if (loop_playback)
            {
                int range = frame_end - frame_start;
                _currentLoopFrame = (_currentLoopFrame + 1) % (range * 2);
                UpdateFrameFromLoop();
                UpdateCurrentFrameData();
            }
            else
            {
                currentFrame++;
                if (currentFrame >= frame_end)
                {
                    if (train && !replay)
                    {
                        if (!enable_standing_reward)
                            EndEpisode();
                        else
                            currentFrame = frame_end - 1;  // 保持末帧，继续学站立
                    }
                    else
                    {
                        currentFrame = frame_end - 1;
                    }
                    UpdateCurrentFrameData();
                }
                else
                {
                    UpdateCurrentFrameData();
                }
            }
        }
        else
        {
            currentFrame++;
            if (currentFrame >= itpData.Count && train && !replay)
            {
                if (!enable_standing_reward) EndEpisode();
            }
            UpdateCurrentFrameData();
        }

        if (tt > 1000 && !enable_standing_reward) EndEpisode();
    }

    // =========================================================
    //  辅助方法
    // =========================================================
    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = stiffness;
        drive.damping = damping;
        drive.target = x;
        joint.xDrive = drive;
    }

    private void UpdateCurrentFrameData()
    {
        if (itpData != null && currentFrame >= 0 && currentFrame < itpData.Count)
        {
            currentData = itpData[currentFrame];
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, 29);
        }
    }

    // =========================================================
    //  公共接口（不变）
    // =========================================================
    public void SwitchModel(string motionName)
    {
        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        List<string> csvFileNames = GetCsvFileNames(streamingAssetsPath);
        for (int i = 0; i < csvFileNames.Count; i++)
        {
            if (Path.GetFileNameWithoutExtension(csvFileNames[i]) == motionName)
            {
                motion_id = i;
                motion_name = motionName;
                LoadAndInterpolateData(csvFileNames[i]);
                return;
            }
        }
    }

    public void SwitchModel(int motionId)
    {
        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        List<string> csvFileNames = GetCsvFileNames(streamingAssetsPath);
        if (motionId >= 0 && motionId < csvFileNames.Count)
        {
            motion_id = motionId;
            LoadAndInterpolateData(csvFileNames[motionId]);
            motion_name = Path.GetFileNameWithoutExtension(csvFileNames[motionId]);
        }
    }

    public void SwitchModel(UnityEngine.Object motionObject)
    {
        if (motionObject != null) SwitchModel(motionObject.name);
    }

    private void LoadAndInterpolateData(string filePath)
    {
        refData = LoadDataFromFile(filePath);
        if (refData.Count > 0)
        {
            float[] refT = new float[refData.Count];
            for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;
            float[] newT = new float[(int)(refData.Count * 100f / 30f) - 5];
            for (int i = 0; i < newT.Length; i++) newT[i] = i / 100f;
            itpData = Interpolate(refT, refData, newT);

            frame_end = Mathf.Min(frame_end, itpData.Count - 1);
            frame_start = Mathf.Min(frame_start, frame_end);
            currentFrame = frame_start;

            // [修改2] 站立参考仍取 itpData[0]
            if (itpData.Count > 0)
            {
                float[] standFrame = itpData[0];
                Array.Copy(standFrame, 7, _standDof, 0, 29);
                _hasStandDof = true;
            }

            UpdateCurrentFrameData();
        }
    }

    public void SetTargetFrame(int frame)
    {
        if (itpData != null && frame >= 0 && frame < itpData.Count)
        {
            currentFrame = frame;
            frame0 = frame;
            UpdateCurrentFrameData();
        }
    }

    public void TeleportToTargetPose()
    {
        newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        newRotation = new Quaternion(-currentRot[1], currentRot[2], currentRot[0], -currentRot[3]);

        arts[0].TeleportRoot(newPosition, newRotation);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;

        float[] Dof = new float[35]{ 0, 0, 0, 0, 0, 0,
            currentDof[12], currentDof[6],  currentDof[0],
            currentDof[13], currentDof[7],  currentDof[1],
            currentDof[14], currentDof[8],  currentDof[2],
            currentDof[15], currentDof[22], currentDof[9],
            currentDof[3],  currentDof[16], currentDof[23],
            currentDof[10], currentDof[4],  currentDof[17],
            currentDof[24], currentDof[11], currentDof[5],
            currentDof[18], currentDof[25], currentDof[19],
            currentDof[26], currentDof[20], currentDof[27],
            currentDof[21], currentDof[28] };

        List<float> jointPositions = new List<float>();
        for (int i = 0; i < 35; i++) jointPositions.Add(Dof[i]);
        arts[0].SetJointPositions(jointPositions);
    }

    public void TeleportToTargetPose(int frame) { SetTargetFrame(frame); TeleportToTargetPose(); }

    public void LoadCSVData(string filePath)
    {
        if (File.Exists(filePath))
        {
            LoadAndInterpolateData(filePath);
            motion_name = Path.GetFileNameWithoutExtension(filePath);
            TeleportToTargetPose();
        }
    }

    public void LoadCSVData(UnityEngine.Object csvObject)
    {
        if (csvObject != null)
            LoadCSVData(Path.Combine(Application.streamingAssetsPath, "g1_dataset", csvObject.name + ".csv"));
    }

    public string GetCurrentMotionName() => motion_name;
    public int GetTotalFrames() => itpData?.Count ?? 0;
    public int GetCurrentFrame() => currentFrame;
    public void SetLoopPlayback(bool loop) => loop_playback = loop;
    public void SetRandomStart(bool random) => rand_start = random;
    public void SetTrainMode(bool isTraining) => train = isTraining;
    public void SetReplayMode(bool isReplay) => replay = isReplay;

    public override void Heuristic(in ActionBuffers actionsOut) { }
}
