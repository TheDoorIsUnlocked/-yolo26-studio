from ultralytics import YOLO


class GetPos:
    result = None

    @classmethod
    def init(cls, source):
        yolo = YOLO(model="icon-V11.pt", task="detect")
        cls.result = yolo(source=source, save=True, conf=0.2, save_txt=True, show=True)

    @classmethod
    def __get_icon_id(cls, icon_name: str):
        for i in cls.result[0].names:
            if cls.result[0].names[i] == icon_name:
                return i
        return None

    @classmethod
    def __get_cls_no(cls, icon_id: int):
        for idx, item in enumerate(cls.result[0].boxes.cls):
            if item == icon_id:
                return idx

    @classmethod
    def __get_xy(cls, cls_no: int):
        x = cls.result[0].boxes.xywh[cls_no][0]
        y = cls.result[0].boxes.xywh[cls_no][1]
        return x, y

    @classmethod
    def get_icon_pos(cls, icon_name: str):
        icon_id = cls.__get_icon_id(icon_name)
        cls_no = cls.__get_cls_no(icon_id)
        pos = cls.__get_xy(cls_no)
        return pos


if __name__ == "__main__":
    GetPos.init(source="001.png")
    overwatch_pos = GetPos.get_icon_pos("overwatch")
    # recyle_bin = GetPos.get_icon_pos('recyle_bin')
    print(overwatch_pos)
