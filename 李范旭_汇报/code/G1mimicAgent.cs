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

public class G1mimicAgent : Agent
{
    public bool train = false;
    public bool replay = false;

    float[] uff = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] u = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] utotal = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int[] trainid = new int[13] { 5, 5, 13, 13, 6, 14, 15, 5, 5, 13, 13, 16, 17 };

    private List<string> actionFolders = new List<string>();
    public int motion_id;
    public string motion_name;

    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();

    private int currentFrame;

    float[] currentData = new float[36];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[29];

    Transform body;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[29];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;
    public int frame0 = 100;

    public float positionKp = 1000f;
    public float positionKd = 50f;
    public float rotationKp = 500f;
    public float rotationKd = 30f;

    private bool _isClone = false;
    private List<string> csvFileNames;
    private string streamingAssetsPath;

    void Start()
    {
        Time.fixedDeltaTime = 0.02f;
        if (train && !_isClone)
        {
            for (int i = 1; i < 34; i++)
            {
                GameObject clone = Instantiate(gameObject);
                clone.transform.position = transform.position + new Vector3(i * 2f, 0, 0);
                clone.name = $"{name}_Clone_{i}";
                clone.GetComponent<G1mimicAgent>()._isClone = true;
            }
        }
    }

    public override void Initialize()
    {
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
        pos0 = body.position;
        rot0 = body.rotation;
        art0.GetJointPositions(P0);
        art0.GetJointVelocities(W0);

        streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "g1_dataset");
        csvFileNames = GetCsvFileNames(streamingAssetsPath);
        if (csvFileNames.Count > 0)
        {
            refData = LoadDataFromFile(csvFileNames[motion_id % csvFileNames.Count]);
            float[] refT = new float[refData.Count];
            for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;
            float[] newT = new float[(int)(refData.Count * 50f / 30f) - 1];
            for (int i = 0; i < newT.Length; i++) newT[i] = i / 50f;
            itpData = Interpolate(refT, refData, newT);
            motion_name = csvFileNames[motion_id % csvFileNames.Count].Replace(Path.Combine(Application.streamingAssetsPath, "g1_dataset") + Path.DirectorySeparatorChar, "").Replace(".csv", "");
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
                        csvFiles.Add(file);
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
                    if (float.TryParse(value.Trim(), out float parsedValue)) frameData.Add(parsedValue);
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
        arts[0].TeleportRoot(pos0, rot0);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;
        arts[0].SetJointPositions(P0);
        arts[0].SetJointVelocities(W0);
        for (int i = 0; i < 29; i++) u[i] = 0;
        for (int i = 0; i < 29; i++) uff[i] = 0;
        currentFrame = frame0;

        if (train)
        {
            if (csvFileNames != null && csvFileNames.Count > 0)
            {
                motion_id = Random.Range(0, csvFileNames.Count);
                refData = LoadDataFromFile(csvFileNames[motion_id]);
                float[] refT = new float[refData.Count];
                for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;
                float[] newT = new float[(int)(refData.Count * 50f / 30f) - 1];
                for (int i = 0; i < newT.Length; i++) newT[i] = i / 50f;
                itpData = Interpolate(refT, refData, newT);
                motion_name = csvFileNames[motion_id].Replace(Path.Combine(Application.streamingAssetsPath, "g1_dataset") + Path.DirectorySeparatorChar, "").Replace(".csv", "");
            }
        }
        else if (replay)
        {
            if (csvFileNames != null && csvFileNames.Count > 0)
            {
                motion_id = (motion_id + 1) % csvFileNames.Count;
                refData = LoadDataFromFile(csvFileNames[motion_id]);
                float[] refT = new float[refData.Count];
                for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;
                float[] newT = new float[(int)(refData.Count * 50f / 30f) - 1];
                for (int i = 0; i < newT.Length; i++) newT[i] = i / 50f;
                itpData = Interpolate(refT, refData, newT);
                motion_name = csvFileNames[motion_id].Replace(Path.Combine(Application.streamingAssetsPath, "g1_dataset") + Path.DirectorySeparatorChar, "").Replace(".csv", "");
            }
        }

        tt = 0;
        if (itpData.Count > frame0)
            currentData = itpData[currentFrame];
        else if (refData.Count > frame0)
            currentData = refData[currentFrame];
        else
            currentData = new float[36];

        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, 29);

        for (int i = 0; i < 29; i++)
        {
            uff[i] = currentDof[i] * 180f / 3.14f;
            SetJointTargetDeg(jh[i], uff[i]);
        }

        Vector3 newPosition = new Vector3(-currentPos[1], currentPos[2] + 0.04f, currentPos[0]);
        Quaternion newRotation = new Quaternion(-currentRot[1], currentRot[2], currentRot[0], -currentRot[3]);
        newPosition.x += pos0.x;
        newPosition.z += pos0.z;

        arts[0].TeleportRoot(newPosition, newRotation);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;
        arts[0].immovable = true;
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
                float[] boundary = new float[dimension];
                Array.Copy(tValue < t[0] ? posList[0] : posList[posList.Count - 1], boundary, dimension);
                result.Add(boundary);
                continue;
            }
            int index = 0;
            while (index < t.Length - 1 && t[index + 1] < tValue)
                index++;
            float ratio = (tValue - t[index]) / (t[index + 1] - t[index]);
            float[] interpolatedPos = new float[dimension];
            for (int j = 0; j < dimension; j++)
                interpolatedPos[j] = Mathf.Lerp(posList[index][j], posList[index + 1][j], ratio);
            result.Add(interpolatedPos);
        }
        return result;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * Mathf.Deg2Rad);
        sensor.AddObservation(EulerTrans(body.eulerAngles[1]) * Mathf.Deg2Rad);
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * Mathf.Deg2Rad);
        sensor.AddObservation(body.InverseTransformDirection(art0.velocity));
        sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        for (int i = 0; i < 29; i++)
        {
            sensor.AddObservation(jh[i].jointPosition[0]);
            sensor.AddObservation(jh[i].jointVelocity[0]);
        }
        for (int i = 0; i < 29; i++)
            sensor.AddObservation(currentDof[i]);
        Vector3 posError = body.InverseTransformPoint(newPosition) - body.InverseTransformPoint(body.position);
        sensor.AddObservation(posError);
        Quaternion rotError = Quaternion.Inverse(body.rotation) * newRotation;
        sensor.AddObservation(rotError.x);
        sensor.AddObservation(rotError.y);
        sensor.AddObservation(rotError.z);
        sensor.AddObservation(rotError.w);
    }

    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180) return eulerAngle;
        else return eulerAngle - 360f;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        float kb = 5.0f;
        if (replay) kb = 0;
        float maxCorrectionDeg = 10f;
        for (int i = 0; i < 29; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i] * maxCorrectionDeg;
            utotal[i] = kb * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }

    void FixedUpdate()
    {
        if (itpData.Count > 0)
        {
            currentData = itpData[currentFrame];
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, 29);
            for (int i = 0; i < 29; i++) uff[i] = currentDof[i] * 180f / 3.14f;

            newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
            newRotation = new Quaternion(-currentRot[1], currentRot[2], currentRot[0], -currentRot[3]);
            newPosition.x += pos0.x;
            newPosition.z += pos0.z;

            if (replay)
            {
                Physics.gravity = Vector3.zero;
                arts[0].TeleportRoot(newPosition, newRotation);
            }
            else
            {
                if (tt > 3)
                {
                    arts[0].immovable = false;
                    Vector3 positionError = newPosition - body.position;
                    Vector3 velocityError = -art0.velocity;
                    Vector3 positionForce = positionKp * positionError + positionKd * velocityError;
                    arts[0].AddForce(positionForce);

                    Quaternion rotationError = newRotation * Quaternion.Inverse(body.rotation);
                    rotationError.ToAngleAxis(out float angle, out Vector3 axis);
                    if (angle > 180f) angle -= 360f;
                    Vector3 rotationErrorVector = (angle * Mathf.Deg2Rad) * axis.normalized;
                    Vector3 angularVelocityError = -art0.angularVelocity;
                    Vector3 rotationTorque = rotationKp * rotationErrorVector + rotationKd * angularVelocityError;
                    art0.AddTorque(rotationTorque);
                }
            }
        }

        tt++;
        float live_reward = 1.0f;
        float rot_reward = 0;
        float pos_reward = 0;
        float dof_reward = 0f;
        float vel_smooth_reward = 0f;
        float angvel_penalty = 0f;

        if (tt > 3)
        {
            arts[0].immovable = false;
            rot_reward = -0.02f * Quaternion.Angle(body.rotation, newRotation);
            pos_reward = -1.0f * (body.position - newPosition).magnitude;
            for (int i = 0; i < 29; i++)
                dof_reward += -0.1f * Mathf.Abs(jh[i].jointPosition[0] - currentDof[i]);
            for (int i = 0; i < 29; i++)
                vel_smooth_reward += -0.005f * Mathf.Abs(jh[i].jointVelocity[0]);
            angvel_penalty = -0.05f * art0.angularVelocity.magnitude;

            // 渐进终止条件
            float progress = Mathf.Clamp01(Academy.Instance.StepCount / 2000000f);
            float angleThreshold = Mathf.Lerp(60f, 25f, progress);
            float posThreshold = Mathf.Lerp(0.8f, 0.3f, progress);

            if (tt > 100 && (Quaternion.Angle(body.rotation, newRotation) > angleThreshold ||
                (body.position - newPosition).magnitude > posThreshold))
            {
                AddReward(-5f);
                EndEpisode();
                return;
            }
        }

        float total_reward = live_reward + rot_reward + pos_reward + dof_reward
                             + vel_smooth_reward + angvel_penalty;
        AddReward(total_reward);

        currentFrame++;
        if (currentFrame >= itpData.Count - 1)
        {
            AddReward(5f);
            EndEpisode();
        }
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f;
        drive.damping = 8f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut) { }
}