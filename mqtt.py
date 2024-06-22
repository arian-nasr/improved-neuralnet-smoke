from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
import sounddevice as sd
import uuid
from scipy.io.wavfile import write
import time
import paho.mqtt.client as mqtt

freq = 16000
chan = 1
duration = 10
bufSize = duration*freq
sd.default.device = 'USB Camera-B3.04.06.1: Audio'

model_path = './model3.tflite'
base_options = core.BaseOptions(file_name=model_path)
classification_options = processor.ClassificationOptions(max_results=2)
options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = audio.AudioClassifier.create_from_options(options)

def on_publish(client, userdata, mid, reason_code, properties):
    # reason_code and properties will only be present in MQTTv5. It's always unset in MQTTv3
    try:
        userdata.remove(mid)
    except KeyError:
        print("on_publish() is called with a mid not present in unacked_publish")
        print("This is due to an unavoidable race-condition:")
        print("* publish() return the mid of the message sent.")
        print("* mid from publish() is added to unacked_publish by the main thread")
        print("* on_publish() is called by the loop_start thread")
        print("While unlikely (because on_publish() will be called after a network round-trip),")
        print(" this is a race-condition that COULD happen")
        print("")
        print("The best solution to avoid race-condition is using the msg_info from publish()")
        print("We could also try using a list of acknowledged mid rather than removing from pending list,")
        print("but remember that mid could be re-used !")

unacked_publish = set()
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_publish = on_publish

mqttc.user_data_set(unacked_publish)
mqttc.connect("192.168.2.103")
mqttc.loop_start()

while True:
    recording = sd.rec(int(bufSize), samplerate=freq, channels=chan)
    sd.wait()

    audio_file = classifier.create_input_tensor_audio()
    audio_file.load_from_array(src=recording)
    audio_result = classifier.classify(audio_file)

    model_output = audio_result.classifications[0].categories[0]

    print(model_output.category_name, model_output.score)

    if model_output.category_name == 'smoke':
        msg_info = mqttc.publish("neural/smoke", "on", qos=1)
        unacked_publish.add(msg_info.mid)
        msg_info.wait_for_publish()
        write('./positives/{}.wav'.format(str(uuid.uuid4())), freq, recording)
        print('saved positive sample')
    
    elif model_output.category_name == 'nosmoke':
        msg_info = mqttc.publish("neural/smoke", "off", qos=1)
        unacked_publish.add(msg_info.mid)
        msg_info.wait_for_publish()
