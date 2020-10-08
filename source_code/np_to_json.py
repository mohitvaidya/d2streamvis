from functools import reduce
import json 


def convert_json(video_id = None, all_preds = None, width = None, height =None, frames_per_second = None, num_frames =None, basepath = None):

    img_pixels = height*width
    duration = num_frames/frames_per_second
    
    print('num_frames is ',num_frames)
    frames = []
    for num_frame, semantic_predictions in enumerate(all_preds):
        # semantic_predictions = next(all_preds)
        # semantic_predictions = item
        objs = []
        for s in semantic_predictions:
            obj = {}
            obj["label"] = s["text"]
            obj['area_percentage'] = float("{0:.2f}".format(s['area']/img_pixels*100))
            obj["score"] = float("{0:.2f}".format(s["score"] if "score" in s else 1))
            objs.append(obj)

        obj_set = {}
        for s in semantic_predictions:
            k = s["text"]
            score = s["score"] if "score" in s else 1
            if not k in obj_set:
                obj_set[k] = {
                    "scores": [score],
                    "areas":  [s["area"]],
                    "label": k
                }
            else:
                obj_set[k]["scores"].append(score)
                obj_set[k]["areas"].append(s["area"])

        u_objs = []
        for k in obj_set:
            u = obj_set[k]
            n = len(u["scores"])
            score_ave = reduce((lambda x, y: x + y), u["scores"])/n
            area_sum = reduce((lambda x, y: x + y), u["areas"])

            obj = {}
            obj["label"] = u["label"]
            obj['area_percentage'] = float("{0:.2f}".format(area_sum/img_pixels*100))
            obj["score"] = float("{0:.2f}".format(score_ave))
            obj["count"] = n
            u_objs.append(obj)
        frame = {
            "frame":num_frame,
            "instances": objs,
            "objects": u_objs,
        }
    
        # print('num_frame is ',total_frames - num_frames + 1)
        # print('num_frame is ',num_frame + 1)
        frames.append(frame)
    data = {
        "video": {
            "meta": {},
            "base_uri": "https://videobank.blob.core.windows.net/videobank",
            "folder": video_id,
            "output-frame-path": "pipeline/detectron2"
        },
        "ml-data": {
            "object-detection": {
                "meta": {'duration':duration, 'fps':frames_per_second,'len_frames':len(frames)},
                "video": {},
                "frames": frames
            }
        }
    }
    return data
    # print(f'writing OD outs inside >>> {basepath}/{video_id}.json')
    # with open(f'{basepath}/{video_id}.json', 'w') as f:
    #     json.dump(data,f)