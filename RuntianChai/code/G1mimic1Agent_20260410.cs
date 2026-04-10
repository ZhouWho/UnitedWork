using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using System;

public class G1mimic1Agent : Agent
{
    public bool train = false;
    public bool replay = false;

    float[] uff = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] u = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] utotal = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    //int[] trainid=new int[13]{3,9, 5,13,6,14,15,   3,9, 5,13,16,17};
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

    //private bool isEndEpisode = false;
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
        //Time.timeScale = 10f;
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
        //print(newT[newT.Length-1]);
        motion_name=csvFileNames[motion_id].Replace("./Assets/Imitation/G1/dataset\\", "").Replace(".csv", "");

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

        if(endT>450)idx=(idx+1)%5;
        if(rand)frame0 = 1000 + Random.Range(0,5)*100;
        //frame0 = 500 + idx*250;
        //frame0 = 500 + Random.Range(0,1001);
        currentFrame = frame0 % itpData.Count;;

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
        //sensor.AddObservation(body.InverseTransformDirection(art0.velocity));
        sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        for (int i = 0; i < 29; i++)
        {
            sensor.AddObservation(jh[i].jointPosition[0]);
            sensor.AddObservation(jh[i].jointVelocity[0]);
        }
        Vector3 epos=body.position-newPosition;
        Vector3 newEuler = newRotation.eulerAngles;
        sensor.AddObservation(epos);
        sensor.AddObservation(newEuler.x);
        sensor.AddObservation(newEuler.z);
        //sensor.AddObservation(currentDof);
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
            // 使用 % 确保索引永远在 0 到 Count-1 之间
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
        }
    }

    void FixedUpdate()
    {
        tt++;

        // ===== 当前状态 =====
        var vel = body.InverseTransformDirection(art0.velocity);
        var angVel = body.InverseTransformDirection(art0.angularVelocity);

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

        // =========================
        // ===== 奖励函数开始 =====
        // =========================

        // ---- 权重（可调）----
        float w_pos = 2.0f;
        float w_rot = 2.0f;
        float w_dof = 1.5f;
        float w_vel = 0.5f;
        float w_energy = 0.001f;

        // ---- 位置误差 ----
        float pos_err = (body.position - newPosition).magnitude;

        // ---- 姿态误差（忽略Y轴）----
        Quaternion bodyRotNoY = Quaternion.Euler(body.eulerAngles.x, 0, body.eulerAngles.z);
        Quaternion refRotNoY = Quaternion.Euler(newRotation.eulerAngles.x, 0, newRotation.eulerAngles.z);
        float rot_err = Quaternion.Angle(bodyRotNoY, refRotNoY) * Mathf.Deg2Rad;

        // ---- 关节误差 + 速度误差 ----
        float dof_err = 0f;
        float vel_err = 0f;

        for (int i = 0; i < 29; i++)
        {
            float q_err = jh[i].jointPosition[0] - currentDof[i];
            float v_err = jh[i].jointVelocity[0];

            dof_err += q_err * q_err;
            vel_err += v_err * v_err;
        }

        // ---- 控制能量（防抖）----
        float energy = 0f;
        for (int i = 0; i < 29; i++)
        {
            energy += u[i] * u[i];
        }

        // ---- 指数奖励（核心改进）----
        float r_pos = Mathf.Exp(-10f * pos_err * pos_err);
        float r_rot = Mathf.Exp(-5f * rot_err * rot_err);
        float r_dof = Mathf.Exp(-2f * dof_err);
        float r_vel = Mathf.Exp(-0.1f * vel_err);
        float r_energy = Mathf.Exp(-energy);

        float reward =
            w_pos * r_pos +
            w_rot * r_rot +
            w_dof * r_dof +
            w_vel * r_vel +
            w_energy * r_energy;

        // normalize（可选但推荐）
        reward /= (w_pos + w_rot + w_dof + w_vel + w_energy);

        AddReward(reward);

        // =========================
        // ===== 终止条件优化 =====
        // =========================

        if (train)
        {
            if (pos_err > 0.7f || rot_err > 1.0f)
            {
                if (!replay)
                {
                    EndEpisode();
                }
            }
        }

        // ===== 帧推进 =====
        currentFrame++;
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f;//2000f;
        drive.damping = 8f;//200f;
        //drive.forceLimit = 300f;//250f;// 33.5f;// 50f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {

    }
}
