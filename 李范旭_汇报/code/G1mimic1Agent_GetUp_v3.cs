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
    public int motion_id = 10;
    public string motion_name;
    public bool rand_start = true;          // ← RSI，默认开启
    public int frame_start = 733;
    public int frame_end = 983;
    public bool loop_playback = false;
    public int frame0 = 0;

    [Header("Episode")]
    public int max_episode_steps = 350;     // 全程约250帧 + 站立保持约100步

    [Header("Reward Weights (和≈1)")]
    public float w_pose = 0.55f;     // 关节姿态匹配（核心）
    public float w_root_pos = 0.15f;     // root 位置跟踪（防爬行漂移）
    public float w_root_height = 0.12f;     // root 高度跟踪（防躺平/弹射）
    public float w_root_rot = 0.08f;     // root 朝向跟踪
    public float w_vel = 0.10f;     // 关节速度匹配

    [Header("Balance Reward (仅站立保持段生效，相位门控)")]
    public float w_balance = 0.30f;     // 站立段额外平衡奖励（竖直 + 静止）
    public float stand_lin_vel_k = 0.5f;    // 越大对 root 线速度越敏感
    public float stand_ang_vel_k = 0.1f;    // 越大对 root 角速度越敏感

    [Header("Termination (跟踪误差)")]
    public float pos_err_terminate = 0.5f;   // 与参考 root 位置误差(米)
    public float height_err_terminate = 0.35f;  // 与参考 root 高度误差(米)
    public int grace_steps = 20;     // 瞬移后等物理稳定再判终止

    [Header("Control")]
    public float actionSmooth = 0.90f;
    public float stiffness = 180f;
    public float damping = 8f;
    public float rlActionGain = 40f;            // kb：训练时全关节都接受 RL 修正

    [Header("Disturbance (起身阶段建议关闭)")]
    public bool add_disturbance = false;
    private float min_disturbance_interval = 2f;
    private float max_disturbance_interval = 5f;
    private float min_disturbance_force = 5f;
    private float max_disturbance_force = 20f;
    private float disturbance_torque = 5f;

    [Header("Debug")]
    public bool debug_getup = true;
    public int debug_interval_steps = 100;

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

    private float _disturbanceTimer = 0f;
    private float _nextDisturbanceInterval = 3f;
    private bool _isClone = false;

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
                if (ActionNum < jh.Length) { jh[ActionNum] = arts[k]; ActionNum++; }
            }
        }

        body = arts[0].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        List<string> csvFileNames = GetCsvFileNames(streamingAssetsPath);

        if (csvFileNames.Count > 0)
        {
            motion_id = Mathf.Clamp(motion_id, 0, csvFileNames.Count - 1);
            LoadAndInterpolateData(csvFileNames[motion_id]);
            motion_name = Path.GetFileNameWithoutExtension(csvFileNames[motion_id]);
            Debug.Log($"[GetUp-v5] Loaded {motion_name}, frames={itpData.Count}, range {frame_start}-{frame_end}, RSI={rand_start}");
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
                if (frameData.Count >= 36) dataList.Add(frameData.ToArray());
            }
        }
        catch (Exception e) { print("Error: " + e.Message); }
        return dataList;
    }

    private void LoadAndInterpolateData(string filePath)
    {
        refData = LoadDataFromFile(filePath);
        if (refData.Count > 0)
        {
            float[] refT = new float[refData.Count];
            for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;

            int newCount = Mathf.Max(1, (int)(refData.Count * 100f / 30f) - 5);
            float[] newT = new float[newCount];
            for (int i = 0; i < newT.Length; i++) newT[i] = i / 100f;

            itpData = Interpolate(refT, refData, newT);

            frame_end = Mathf.Min(frame_end, itpData.Count - 1);
            frame_start = Mathf.Clamp(frame_start, 0, frame_end);
            currentFrame = frame_start;
            UpdateCurrentFrameData();
        }
    }

    List<float[]> Interpolate(float[] t, List<float[]> posList, float[] targetT)
    {
        List<float[]> result = new List<float[]>();
        if (posList.Count == 0) return result;
        int dimension = posList[0].Length;
        for (int i = 0; i < targetT.Length; i++)
        {
            float tValue = targetT[i];
            int index = 0;
            while (index < t.Length - 2 && t[index + 1] < tValue) index++;
            float denom = t[index + 1] - t[index];
            float ratio = Mathf.Abs(denom) < 1e-6f ? 0f : (tValue - t[index]) / denom;
            float[] interpolatedPos = new float[dimension];
            for (int j = 0; j < dimension; j++)
                interpolatedPos[j] = Mathf.Lerp(posList[index][j], posList[index + 1][j], ratio);
            result.Add(interpolatedPos);
        }
        return result;
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
    //  OnEpisodeBegin —— 含 RSI
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

        if (frame_end > frame_start)
            currentFrame = rand_start ? Random.Range(frame_start, frame_end) : frame_start;
        else
            currentFrame = frame0;

        tt = 0;
        UpdateCurrentFrameData();

        newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        newRotation = new Quaternion(-currentRot[1], currentRot[2], currentRot[0], -currentRot[3]);

        arts[0].TeleportRoot(newPosition, newRotation);
        foreach (ArticulationBody ab in arts)
            if (ab != null) { ab.velocity = Vector3.zero; ab.angularVelocity = Vector3.zero; }

        ApplyCurrentDofToRobot();
        ResetJointVelocities();
    }

    void ApplyCurrentDofToRobot()
    {
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

    void ResetJointVelocities()
    {
        try
        {
            List<float> jv = new List<float>();
            for (int i = 0; i < 35; i++) jv.Add(0f);
            arts[0].SetJointVelocities(jv);
        }
        catch { }
    }

    // =========================================================
    //  CollectObservations —— 维度 = 128
    // =========================================================
    public override void CollectObservations(VectorSensor sensor)
    {
        // 躯干姿态 (2) + 角速度 (3)
        sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * Mathf.PI / 180f);
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * Mathf.PI / 180f);
        sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity)); // 3

        // 关节位置 (29) + 关节速度 (29)
        for (int i = 0; i < 29; i++) sensor.AddObservation(jh[i].jointPosition[0]);
        for (int i = 0; i < 29; i++) sensor.AddObservation(jh[i].jointVelocity[0]);

        // 当前帧参考关节角 (29)
        for (int i = 0; i < 29; i++) sensor.AddObservation(currentDof[i]);

        // 下一帧参考速度 (29)
        if (currentFrame + 1 < itpData.Count)
        {
            Array.Copy(itpData[currentFrame + 1], 7, nextDof, 0, 29);
            for (int i = 0; i < 29; i++) sensor.AddObservation((nextDof[i] - currentDof[i]) * 100f);
        }
        else
        {
            for (int i = 0; i < 29; i++) sensor.AddObservation(0f);
        }

        // root 位置误差 (3) + 参考朝向 x/z (2)
        Vector3 epos = body.position - newPosition;
        Vector3 newEuler = newRotation.eulerAngles;
        sensor.AddObservation(epos);
        sensor.AddObservation(newEuler.x);
        sensor.AddObservation(newEuler.z);

        // [v5 新增] 归一化相位 φ (1) + 参考 root 高度 (1)
        float phase = (float)(currentFrame - frame_start) / Mathf.Max(1, frame_end - frame_start);
        sensor.AddObservation(phase);
        sensor.AddObservation(newPosition.y);
    }

    float EulerTrans(float angle)
    {
        angle = angle % 360f;
        if (angle > 180f) angle -= 360f;
        else if (angle < -180f) angle += 360f;
        return angle;
    }

    // =========================================================
    //  OnActionReceived —— 含腿部 kb 修复
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
                foreach (ArticulationBody ab in arts)
                    if (ab != null) { ab.velocity = Vector3.zero; ab.angularVelocity = Vector3.zero; }
                ApplyCurrentDofToRobot();
                ResetJointVelocities();
            }
        }

        var continuousActions = actionBuffers.ContinuousActions;
        for (int i = 0; i < 29; i++)
        {
            u[i] = u[i] * actionSmooth + (1f - actionSmooth) * continuousActions[i];
            float kb_i = replay ? 0f : rlActionGain;   // 全关节（含腿部）都接受 RL 修正
            utotal[i] = kb_i * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
            action_filtered[i] = u[i];
        }
    }

    // =========================================================
    //  FixedUpdate —— 纯模仿奖励 + 误差终止
    // =========================================================
    void FixedUpdate()
    {
        tt++;

        if (add_disturbance)
        {
            _disturbanceTimer += Time.fixedDeltaTime;
            if (_disturbanceTimer >= _nextDisturbanceInterval)
            {
                _disturbanceTimer = 0f;
                _nextDisturbanceInterval = Random.Range(min_disturbance_interval, max_disturbance_interval);
                float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
                Vector3 dir = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
                art0.AddForce(dir * Random.Range(min_disturbance_force, max_disturbance_force), ForceMode.Impulse);
                if (disturbance_torque > 0)
                    art0.AddTorque(Random.insideUnitSphere.normalized * disturbance_torque, ForceMode.Impulse);
            }
        }

        if (tt > 1)
        {
            float reward = ComputeImitationReward();
            AddReward(reward);

            bool ended = CheckTermination();
            if (!ended)
            {
                AdvanceFrame();

                if (tt >= max_episode_steps && train && !replay)
                    EndEpisode();   // 纯模仿，不加额外惩罚
            }
        }
        else
        {
            AdvanceFrame();
        }

        Array.Copy(action_filtered, action_prev, 29);
    }

    float ComputeImitationReward()
    {
        // 1) 关节姿态
        float pose_reward = 0f;
        for (int i = 0; i < 29; i++)
        {
            float e = jh[i].jointPosition[0] - currentDof[i];
            pose_reward += Mathf.Exp(-5f * e * e);
        }
        pose_reward /= 29f;

        // 2) 关节速度
        float vel_reward = 1f;
        if (currentFrame + 1 < itpData.Count)
        {
            vel_reward = 0f;
            Array.Copy(itpData[currentFrame + 1], 7, nextDof, 0, 29);
            for (int i = 0; i < 29; i++)
            {
                float targetVel = (nextDof[i] - currentDof[i]) * 100f;
                float ve = jh[i].jointVelocity[0] - targetVel;
                vel_reward += Mathf.Exp(-0.015f * ve * ve);
            }
            vel_reward /= 29f;
        }

        // 3) root 位置（全3D，锚定水平、防爬行漂移）
        float posErr = (body.position - newPosition).magnitude;
        float root_pos_reward = Mathf.Exp(-2f * posErr * posErr);

        // 4) root 高度（跟踪参考高度，防躺平/弹射）
        float heightErr = body.position.y - newPosition.y;
        float root_height_reward = Mathf.Exp(-10f * heightErr * heightErr);

        // 5) root 朝向
        Vector3 be = body.eulerAngles;
        Vector3 re = newRotation.eulerAngles;
        float rotErr = (Mathf.Abs(EulerTrans(be.x) - EulerTrans(re.x)) +
                        Mathf.Abs(EulerTrans(be.z) - EulerTrans(re.z))) * Mathf.PI / 360f;
        float root_rot_reward = Mathf.Exp(-2f * rotErr * rotErr);

        float reward = w_pose * pose_reward
                     + w_vel * vel_reward
                     + w_root_pos * root_pos_reward
                     + w_root_height * root_height_reward
                     + w_root_rot * root_rot_reward;

        // [v5.1] 站立保持段的平衡奖励（相位门控：仅当参考已到站立帧时启用）
        // 躺/起身阶段身体是横的，不能奖励"竖直"，否则会逼它提前乱站
        bool inStandHold = (frame_end > frame_start) && (currentFrame >= frame_end - 1);
        float balance_reward = 0f;
        if (inStandHold)
        {
            float uprightDot = Vector3.Dot(body.up, Vector3.up);     // 站直≈1，躺≈0
            float upright = Mathf.Clamp01(uprightDot);
            Vector3 v = art0.velocity;
            Vector3 w = art0.angularVelocity;
            float stillness = Mathf.Exp(-stand_lin_vel_k * v.sqrMagnitude
                                        - stand_ang_vel_k * w.sqrMagnitude);
            balance_reward = w_balance * (0.5f * upright + 0.5f * stillness);
            reward += balance_reward;
        }

        if (debug_getup && !_isClone && tt % Mathf.Max(1, debug_interval_steps) == 0)
        {
            Debug.Log($"[GetUp-v5.1] t={tt}, frame={currentFrame}, reward={reward:F3}, " +
                      $"rootY={body.position.y:F3}, refY={newPosition.y:F3}, " +
                      $"posErr={posErr:F2}, hErr={heightErr:F2}, pose={pose_reward:F2}, " +
                      $"stand={(inStandHold ? 1 : 0)}, bal={balance_reward:F2}");
        }

        return reward;
    }

    bool CheckTermination()
    {
        if (!train || replay) return false;
        if (tt <= grace_steps) return false;

        float posErr = (body.position - newPosition).magnitude;
        float heightErr = Mathf.Abs(body.position.y - newPosition.y);

        if (posErr > pos_err_terminate || heightErr > height_err_terminate)
        {
            if (debug_getup && !_isClone)
                Debug.Log($"[GetUp-v5] TERMINATE t={tt}, frame={currentFrame}, posErr={posErr:F2}, hErr={heightErr:F2}");
            EndEpisode();
            return true;
        }
        return false;
    }

    void AdvanceFrame()
    {
        if (frame_end > frame_start)
        {
            currentFrame++;
            if (currentFrame >= frame_end)
                currentFrame = frame_end - 1;   // 到达末尾后保持站立参考帧（学"站稳并保持"）
            UpdateCurrentFrameData();
        }
        else
        {
            currentFrame++;
            if (currentFrame >= itpData.Count) currentFrame = Mathf.Max(0, itpData.Count - 1);
            UpdateCurrentFrameData();
        }
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = stiffness;
        drive.damping = damping;
        drive.target = x;
        joint.xDrive = drive;
    }

    // =========================================================
    //  公共接口（切换动作用，可选）
    // =========================================================
    public void SwitchModel(int motionId)
    {
        string p = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        List<string> csv = GetCsvFileNames(p);
        if (motionId >= 0 && motionId < csv.Count)
        {
            motion_id = motionId;
            LoadAndInterpolateData(csv[motionId]);
            motion_name = Path.GetFileNameWithoutExtension(csv[motionId]);
        }
    }

    public string GetCurrentMotionName() => motion_name;
    public int GetTotalFrames() => itpData?.Count ?? 0;
    public int GetCurrentFrame() => currentFrame;
    public void SetRandomStart(bool r) => rand_start = r;
    public void SetTrainMode(bool t) => train = t;
    public void SetReplayMode(bool r) => replay = r;

    public override void Heuristic(in ActionBuffers actionsOut) { }
}
