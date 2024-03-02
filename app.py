import os, stat
import subprocess
from zipfile import ZipFile
import time
import torch
import torchaudio

os.system('python -m unidic download')

import re
import gradio as gr
from scipy.io.wavfile import write
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from huggingface_hub import HfApi

os.environ["COQUI_TOS_AGREED"] = "1"
HF_TOKEN = os.environ.get("HF_TOKEN")

api = HfApi(token=HF_TOKEN)
repo_id = "coqui/xtts"

print("Export newer ffmpeg binary for denoise filter")
ZipFile("ffmpeg.zip").extractall()
print("Make ffmpeg binary executable")
st = os.stat("ffmpeg")
os.chmod("ffmpeg", st.st_mode | stat.S_IEXEC)

# This will trigger downloading model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS downloaded")

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=True,
)
model.cuda()

# This is for debugging purposes only
DEVICE_ASSERT_DETECTED = 0
DEVICE_ASSERT_PROMPT = None
DEVICE_ASSERT_LANG = None

supported_languages = config.languages

voices = {
    "lula": "./files/lula.mp3",
    "bolsonaro": "./files/bolsonaro.mp3",
    "away": "./files/away.mp3",
    "faustao": "./files/faustao.mp3",
    "silvio_santos": "./files/silvio_santos.mp3",
}

print("pre processing audio steps")

for voiceKey in list(voices.keys()):
    current_audio_path = voices[voiceKey]
    print(f"processing {voiceKey}")
    try:
        out_filename = (
            current_audio_path + ".wav"
        )
        voices[voiceKey] = out_filename
        shell_command = f"./ffmpeg -y -i {current_audio_path} -af lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02, {out_filename}".split(
            " "
        )
        command_result = subprocess.run(
            [item for item in shell_command],
            capture_output=False,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Error: failed filtering, use original microphone input")

def predict(
    prompt,
    voice,
):
        language = "pt"
        audio_file_pth = voices[voice]

        speaker_wav = audio_file_pth

        if len(prompt) < 1:
            return (
                None,
                None,
                None,
                None,
            )
        if len(prompt) > 200:
            gr.Warning(
                "Text length limited to 200 characters for this demo, please try shorter text. You can clone this space and edit code for your own usage"
            )
            return (
                None
            )
        global DEVICE_ASSERT_DETECTED
        if DEVICE_ASSERT_DETECTED:
            global DEVICE_ASSERT_PROMPT
            global DEVICE_ASSERT_LANG
            # It will likely never come here as we restart space on first unrecoverable error now
            print(
                f"Unrecoverable exception caused by language:{DEVICE_ASSERT_LANG} prompt:{DEVICE_ASSERT_PROMPT}"
            )

            # HF Space specific.. This error is unrecoverable need to restart space
            space = api.get_space_runtime(repo_id=repo_id)
            if space.stage!="BUILDING":
                api.restart_space(repo_id=repo_id)
            else:
                print("TRIED TO RESTART but space is building")

        try:
            metrics_text = ""
            t_latent = time.time()

            # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
            try:
                (
                    gpt_cond_latent,
                    speaker_embedding,
                ) = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
            except Exception as e:
                print("Speaker encoding error", str(e))
                gr.Warning(
                    "It appears something wrong with reference, did you unmute your microphone?"
                )
                return (
                    None,
                )

            # temporary comma fix
            prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",prompt)
            
            #print("I: Generating new audio...")
            t0 = time.time()
            out = model.inference(
                prompt,
                language,
                gpt_cond_latent,
                speaker_embedding,
                repetition_penalty=5.0,
                temperature=0.75,
            )
            #inference_time = time.time() - t0
            #print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
            #metrics_text+=f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
            #real_time_factor= (time.time() - t0) / out['wav'].shape[-1] * 24000
            #print(f"Real-time factor (RTF): {real_time_factor}")
            #metrics_text+=f"Real-time factor (RTF): {real_time_factor:.2f}\n"
            torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

            return "output.wav"

        except RuntimeError as e:
            if "device-side assert" in str(e):
                gr.Warning("Unhandled Exception encounter, please retry in a minute")
                print("Cuda device-assert Runtime encountered need restart")

                space = api.get_space_runtime(repo_id=repo_id)
                if space.stage!="BUILDING":
                    api.restart_space(repo_id=repo_id)
                else:
                    print("TRIED TO RESTART but space is building")

            else:
                if "Failed to decode" in str(e):
                    print("Speaker encoding error", str(e))
                    gr.Warning(
                        "It appears something wrong with reference, did you unmute your microphone?"
                    )
                else:
                    print("RuntimeError: non device-side assert error:", str(e))
                    gr.Warning("Something unexpected happened please retry again.")
                return (
                    None,
                )

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Texto",
                value="Opa",
            )
            voice = gr.Dropdown(
                label="Voz",
                choices=list(voices.keys()),
                max_choices=1,
                value="lula",
            )

            tts_button = gr.Button("Processar", elem_id="send-btn", visible=True)


        with gr.Column():
            audio_gr = gr.Audio(label="Synthesised Audio")


    tts_button.click(predict, [input_text_gr, voice], outputs=[audio_gr])

demo.queue(concurrency_count=4, max_size=200)  
demo.launch(debug=True, show_api=True, share=True)
