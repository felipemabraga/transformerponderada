# transformerponderada

## Observação 1

<img width="1505" height="91" alt="image" src="https://github.com/user-attachments/assets/5b5399b7-d471-404f-938d-a9721615d3bb" />

- **Duração por step:** cada passo está levando ~21s, o que indica execução muito lenta (provavelmente tem algum processo em CPU, não em GPU).  
- **Estimativa de tempo total:** o log aponta 4h41min restantes só para a primeira época, sugerindo que 3 épocas completas levariam mais de 12h. Isso reforça a diferença CPU vs GPU que você precisa documentar.  
- **Loss inicial:** `8.8892`, valor esperado no começo (modelo ainda sem aprendizado significativo).  
- **masked_accuracy:** `0.0`, também normal no início, pois o modelo ainda não captou padrões.  

##Observação 2 - Benchmarking GPU e CPU

---

## Código para Benchmark (CPU vs GPU)

```python
import time
import tensorflow as tf

def bench_fit_simple(model, train_ds, steps=10, device="/GPU:0"):
    """Treina 1 época por poucos steps e retorna segundos por step."""
    with tf.device(device):
        t0 = time.perf_counter()
        model.fit(train_ds, epochs=1, steps_per_epoch=steps, verbose=0)
        t1 = time.perf_counter()
    return (t1 - t0) / steps

# Use um subset pequeno para acelerar
small_train = train_batches.take(50).prefetch(tf.data.AUTOTUNE)

# GPU
gpu_sps = bench_fit_simple(transformer, small_train, steps=10, device="/GPU:0")

# CPU
cpu_sps = bench_fit_simple(transformer, small_train, steps=10, device="/CPU:0")

print(f"GPU ~ seg/step: {gpu_sps:.3f}")
print(f"CPU ~ seg/step: {cpu_sps:.3f}")

