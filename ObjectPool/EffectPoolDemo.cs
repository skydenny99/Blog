using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace EffectPool
{ 

    public class EffectPoolDemo : MonoBehaviour
    {
        // Start is called before the first frame update
        void Start()
        {
        
        }

        // Update is called once per frame
        void Update()
        {
            if(Input.GetKeyDown(KeyCode.T))
            {
                var g = EffectPool.Instance.GetEffect("Test");
                g.PlayEffect();
                g.Effect.transform.position.Set(0, 0, 0);
            }
        }

    }
}
