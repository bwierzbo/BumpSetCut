**Volleyball Tracking**

How to use
1. Get a couple videos of volleyball gameplay about a minute per video should suffice for getting data for the model
2. Get detected circle images (make sure to change path to video of volleyball gameplay)

*python getball.py*

4. Manually classify the images into two folders ball/notBall
5. Format images and traintestsplit

*python formattraintestsplit.py*

7. Create keras model from dataset created

*python train.py*

8. Run playback with ball detection

*python isBall.py*
