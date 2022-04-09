using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CoroutineCache : MonoBehaviour
{
    private WaitForSeconds _cache;
    private float _time;
    private System.DateTime _real_time;

    void Start()
    {
        Application.targetFrameRate = 60;
        _cache = new WaitForSeconds(.1f);
        _time = Time.time;
        _real_time = System.DateTime.Now;
        for (int i = 0; i < 100000; i++)
            StartCoroutine(PrintLogWithoutCache(i));

    }

    IEnumerator PrintLogWithCache(int num)
    {
        for (int i = 0; i < 100; i++)
            yield return _cache;
        if(num == 0)
        {
            Debug.Log("Cache Time : " + (Time.time - _time));
            Debug.Log("Cache Real Time : " + (System.DateTime.Now - _real_time));
        }
    }

    IEnumerator PrintLogWithoutCache(int num)
    {
        for (int i = 0; i < 100; i++)
            yield return new WaitForSeconds(.1f);
        if (num == 0)
        {
            Debug.Log("No Cache Time : " + (Time.time - _time));
            Debug.Log("No Cache Real Time : " + (System.DateTime.Now - _real_time));
        }
    }
}

public class CoroutineCacheDictionary
{
    public static Dictionary<float, WaitForSeconds> _waitForSecondsCache;
    public static WaitForSeconds GetWaitForSecondsCache(float t)
    {
        if (_waitForSecondsCache.ContainsKey(t))
            return _waitForSecondsCache[t];

        var tmp = new WaitForSeconds(t);
        _waitForSecondsCache.Add(t, tmp);
        return tmp;
    }


    public static Dictionary<Func<bool>, WaitUntil> _waitUntilCache;
    public static WaitUntil GetWaitUntilCache(Func<bool> func)
    {
        if (_waitUntilCache.ContainsKey(func))
            return _waitUntilCache[func];

        var tmp = new WaitUntil(func);
        _waitUntilCache.Add(func, tmp);
        return tmp;
    }
}
