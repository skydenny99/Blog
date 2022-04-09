using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace EffectPool
{

    public class EffectTimer : MonoBehaviour, IPoolableEffect
    {
        [SerializeField] private string effectName;
        public string EffectName { get => effectName; }
        public GameObject Effect { get => gameObject; }

        [SerializeField] private float timeout;

        private float startTime;

        private IEnumerator Timer()
        {
            while (IsAlive())
                yield return null;

            EffectPool.Instance.ReleaseEffect(this);
        }

        public bool IsAlive()
        {
            return (startTime + timeout) > Time.time;
        }

        public void PlayEffect()
        {
            gameObject.SetActive(true);
            startTime = Time.time;
            StartCoroutine(Timer());
            Debug.Log(string.Format("Play Effect\n Name: {0}, InstanceID: {1}", effectName, gameObject.GetInstanceID()));
        }

        public void Reset()
        {
            Debug.Log(string.Format("Reset Effect\n Name: {0}, InstanceID: {1}", effectName, gameObject.GetInstanceID()));
        }


    }
    
}