<img width="1750" height="554" alt="image" src="https://github.com/user-attachments/assets/37d9b73f-700e-4cdd-85c6-8211b4a62ae4" />

Business case:
- Mobile application for tractor parts classification

Solution:
- MobileNet was trained based on small dataset, to classify filters.
- Telegram bot was used as a user interface for getting image and sending prediction back.

Deploy:
 - Virtual machine was used to run container's with bot and model. Image's were upload to VM from DockerHub
 - 
Features:
- Trained model and settings, as well as other functionalities, are packed to image's and post to DockerHub
- DockerHub link: docker pull livinmax/telebot-mobilenet-model_service:1.0
- DockerHub link: docker pull livinmax/telebot-mobilenet-bot_service:1.0
- Code for model training in file train_model.py

Project structure:
- bot_service
-   bot.py
-   Dockerfile
-   Dockerfile
-  model_service
-    app.py
-    Dockerfile
-    mobilenetv2_4class.pth
-    requirements.txt
- .env
- docker-compose.yml
