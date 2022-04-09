using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace EffectPool
{

    public class EffectDatas : ScriptableObject
    {
        public List<GameObject> EffectList;
        private Dictionary<string, GameObject> effectDictionary;
        private void OnValidate()
        {
            effectDictionary = new Dictionary<string, GameObject>();
            foreach (var g in EffectList)
            {
                if (g.TryGetComponent<IPoolableEffect>(out var poolableEffect))
                {
                    effectDictionary.Add(poolableEffect.EffectName, g);
                }
            }
        }

        public IPoolableEffect FindEffect(string key)
        {
            if (effectDictionary.TryGetValue(key, out var value))
            {
                if (value.TryGetComponent<IPoolableEffect>(out var effect))
                    return effect;
                Debug.LogError(string.Format("{0} don't have IPoolableEffect script", key));
            }
            return null;
        }
    }

}