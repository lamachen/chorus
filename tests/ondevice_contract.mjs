import { AutoProcessor, RawImage, env } from '@huggingface/transformers';
env.allowLocalModels = false;
const MODEL = 'onnx-community/FastVLM-0.5B-ONNX';
const processor = await AutoProcessor.from_pretrained(MODEL);
console.log('processor ok:', processor.constructor.name);
const W=384,H=384; const data=new Uint8ClampedArray(W*H*4);
for(let i=0;i<data.length;i+=4){data[i]=100;data[i+1]=70;data[i+2]=160;data[i+3]=255;}
const img=new RawImage(data,W,H,4);
const messages=[
 {role:'system',content:'ROLLE: Du bist Mara, 24, aus Berlin, ein echter Mensch in einer Random-Video-Chat-App. Deine Kamera ist kaputt oder aus, dein Mikro geht, du SIEHST das Video deines Gegenuebers und HOERST ihn. Stil: gesprochene Sprache, 1-3 kurze Saetze, max 30 Woerter, keine Listen, kein Markdown, keine Regieanweisungen. VERBOTEN: zugeben eine KI zu sein; Assistenten-Floskeln. Antworte NUR mit dem, was du laut sagst.'},
 {role:'user',content:'Hallo! Was siehst du gerade bei mir?'},
 {role:'assistant',content:'Hey! Bei dir ist es ja richtig dunkel, nur so blaues Licht.'},
 {role:'user',content:'<image>\nUnd jetzt, was siehst du jetzt?'}];
const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
console.log('template ok, laenge', prompt.length, '| ende:', JSON.stringify(prompt.slice(-60)));
const inputs = await processor(img, prompt, { add_special_tokens: false });
console.log('input-keys:', Object.keys(inputs).join(', '));
console.log('input_ids', JSON.stringify(inputs.input_ids.dims), '| pixel_values', JSON.stringify(inputs.pixel_values.dims));
const ids = inputs.input_ids.data;
console.log('enthält image-token-id -200?', Array.from(ids).includes(-200) || 'nein (processor mappt selbst)');
