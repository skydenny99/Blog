using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace EffectPool
{
    public interface IPoolableEffect
    {
        public string EffectName { get; }
        public GameObject Effect { get; }

        public void PlayEffect();
        public bool IsAlive();
        public void Reset();
    }


    public class EffectPool : MonoBehaviour
    {

        private static EffectPool _instance;
        public static EffectPool Instance => _instance;

        [SerializeField] private EffectDatas datas;

        private Dictionary<string, List<IPoolableEffect>> effectPool;


        private void Awake()
        {
            _instance = this;
            effectPool = new Dictionary<string, List<IPoolableEffect>>();
        }

        public void ReleaseEffect(IPoolableEffect effect)
        {
            if(effectPool.TryGetValue(effect.EffectName, out var poolableEffects))
            {
                effect.Reset();
                effect.Effect.SetActive(false);
                poolableEffects.Add(effect);
            }
            else
            {
                var list = new List<IPoolableEffect>();
                effectPool.Add(effect.EffectName, list);
                list.Add(effect);
            }
        }

        public IPoolableEffect GetEffect(string key)
        {
            if(effectPool.TryGetValue(key, out var poolableEffects))
            {
                Debug.Log("Get Effect name: " + key);
                if(poolableEffects.Count > 0)
                {
                    var effect = poolableEffects[0];
                    poolableEffects.RemoveAt(0);
                    return effect;
                }
                else
                {
                    var effect = MakeEffect(key);
                    return effect;
                }
            }
            else
            {
                var effect = MakeEffect(key);
                effectPool.Add(effect.EffectName, new List<IPoolableEffect>());
                return effect;
            }

        }

        private IPoolableEffect MakeEffect(string key)
        {
            Debug.Log("Make Effect name: " + key);
            var effect = datas.FindEffect(key);
            if(effect != null)
            {
                var clone = Instantiate(effect.Effect);
                clone.SetActive(false);
                return clone.GetComponent<IPoolableEffect>();
            }
            Debug.LogError("Make Effect Failed! : " + key);
            return null;
        }

    }

    

    public class MakeScriptableObject
    {
        [MenuItem("Assets/Create/EffectDatas")]
        public static void CreateMyAsset()
        {
            var asset = ScriptableObject.CreateInstance<EffectDatas>();

            AssetDatabase.CreateAsset(asset, "Assets/NewScripableObject.asset");
            AssetDatabase.SaveAssets();

            EditorUtility.FocusProjectWindow();

            Selection.activeObject = asset;
        }
    }
}