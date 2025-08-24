# transformerponderada


1) Observação 1:
<img width="1505" height="91" alt="image" src="https://github.com/user-attachments/assets/5b5399b7-d471-404f-938d-a9721615d3bb" />
- Duração por step: cada passo está levando ~21s, o que indica execução muito lenta (provavelmente tem algum processo em CPU, não em GPU).
- Estimativa de tempo total: o log aponta 4h41min restantes só para a primeira época, sugerindo que 3 épocas completas levariam mais de 12h. Isso    reforça a diferença CPU vs GPU que você precisa documentar.
- Loss inicial: 8.8892, valor esperado no começo (modelo ainda sem aprendizado significativo).
- masked_accuracy: 0.0, também normal no início, pois o modelo ainda não captou padrões.

