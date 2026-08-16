from tools_call_deepseek import access_deepseek
import tools
from tools import (
    CHARACTER_TEMPLATE,
    create_name_to_filename_mapping,
    find_and_extract_summaries,
    extract_dicts_with_number,
)
from typing import Dict, Any
import os
import json
import re
from config import (
    GERERATE_PROMPT_PATH,
    NOVEL_STYLES_FOLDER,
    DEFAULT_PROJECT,
    ensure_project_dirs,
    get_project_paths,
)


TEMPLATE = CHARACTER_TEMPLATE


def writting_novel(last_passage: str,
                   this_passage_request: str,
                   article_name: str,
                   chapter_number: str,
                   chapter_name: str,
                   characters: list,
                   world: str,
                   styles: str,
                   words: int,
                   is_NPC: bool,
                   project: str = DEFAULT_PROJECT) -> str:

    ensure_project_dirs(project)
    paths = get_project_paths(project)
    novel_folder = paths["novel"]
    characters_folder = paths["characters"]
    worlds_folder = paths["worlds"]

    history = find_and_extract_summaries(novel_folder, int(chapter_number))
    history_str = json.dumps(history, ensure_ascii=False, indent=2)

    result_mapping = create_name_to_filename_mapping(characters_folder)

    character_history_list = []
    for character in characters:
        character_json_name = result_mapping[character] + ".json"
        character_json_path = os.path.join(characters_folder, character_json_name)
        character_info = tools.load_json(character_json_path)

        character_name = character_info["name"]
        role_introduction = character_info["role_introduction"]
        role_experience = character_info["role_experience"]
        role_personality = character_info["role_personality"]
        plot_history = character_info["plot_history"]

        plot_history = extract_dicts_with_number(plot_history, int(chapter_number))
        plot_history_str = json.dumps(plot_history, ensure_ascii=False, indent=2)

        character_history_list.append(TEMPLATE.format(character_name = character_name,
                                                       role_introduction = role_introduction,
                                                       role_experience = role_experience,
                                                       role_personality = role_personality,
                                                       plot_history = plot_history_str))

    character_history_list_str = json.dumps(character_history_list, ensure_ascii=False, indent=2)

    PROMPT = tools.read_txt_file(GERERATE_PROMPT_PATH)
    FINAL_PROMPT = PROMPT.format(last_passage = last_passage,
                                 this_passage_request = this_passage_request,
                                 history = history_str,
                                 characters = character_history_list_str,
                                 world = world,
                                 styles = styles,
                                 words = words,
                                 is_NPC = is_NPC)
    passage, tokens = access_deepseek(FINAL_PROMPT, caller="writting_novel", return_tokens=True)
    print(f"[Token] 小说生成 → prompt: {tokens['prompt_tokens']}, completion: {tokens['completion_tokens']}, total: {tokens['total_tokens']}")

    return passage





if __name__ == "__main__":
    
    last_passage = """
\n　　
　　贝拉吸了吸鼻子，转身便挎起藤篮，脚步匆匆地走向山坡上那片日益繁茂的药圃，那里新生的甘菊和金盏花正等着采摘晾晒。欧斯默默解下背上的硬木长弓，手指习惯性地拂过紧绷的弓弦，开始了每天例行化的巡逻任务。克罗埃低着头赶路，今天他还有几家伤患等待着他的处理。
　　
　　清晨微凉的空气带着泥土和草叶的清新气息，被他们急促的脚步搅动。村庄刚刚苏醒的轮廓在薄雾中显得柔和，屋顶升起几缕淡白的炊烟，与他们各自奔赴的方向交织在一起，无声宣告着新一天的责任开始。
　　
　　仿佛大家已经开始接受了分别。
　　
　　伙伴们奔忙的轨迹清晰地烙印在村庄逐渐苏醒的脉络里，如同溪流沿着既定的河床奔涌。
　　
　　贝拉与克罗埃每日穿梭于村落间，抚平村民的伤痛，欧斯的箭矢也在村界边缘划出沉默的守护线。唯有我，仿佛被遗留在时光湍急的岸边，像一个长不大的孩子。
　　
　　晨光熹微中，木桩依旧承受着剑锋单调而执拗的撞击，汗水沿着额角滑落，刺痛眼角。每一次挥臂，每一次格挡带来的震动沿着剑柄传遍手臂，唤醒的不再是旧伤的隐痛，而是一种啃噬骨髓的空洞焦灼。
　　
　　伙伴们的声音和身影在劳作中变得具体且充满力量，渐渐地成为了这个村庄的一部分。
　　
　　贝拉治愈发热孩童时疲惫却欣慰的笑容，欧斯巡逻归来时沾满尘土的靴子和沉稳的眼神，村民在克罗埃小屋前低声的感激话语。这些景象如同细密的芒刺，扎进我日复一日单调且空洞的练习循环里。
　　
　　我像是一块无法嵌入齿轮组的顽石，只能徒劳地看着精密的机器在身旁高效运转。每一次木剑撞击的闷响，都像是在叩问此刻所作所为的意义。

　　太阳高悬，灼烤着院子里干硬的泥地。汗水浸透了我的粗布上衣，紧贴在脊背上，带来一阵黏腻的烦躁。我停下机械的挥剑动作，木剑沉重地垂在身侧，剑尖在滚烫的泥地上戳出一个小坑。一种巨大的空虚感如同藤蔓，缠绕住四肢百骸，勒得我几乎喘不过气。
　　
　　这看似规律且充实的练习，在伙伴们为村庄付出的具体贡献面前，显得如此苍白、如此……悠闲得可耻。
　　
　　一股无名火猛地窜起，烧得喉咙发干。我狠狠将木剑掼在地上，剑身砸起一小片干燥的尘土，发出沉闷的响声。焦躁在胸膛里冲撞，像一头困兽，找不到出口。
　　
　　午后的寂静被木剑落地的声音粗暴地撕裂，几只原本在篱笆上梳理羽毛的麻雀惊叫着飞远。院子里只剩下我粗重的喘息和汗水滴落尘土时细微的声响。那根被反复劈砍的木桩，表面布满了新旧交叠的剑痕，沉默地矗立着，像是对我所有疑问和愤怒的无言嘲讽。
　　
　　这日复一日的挥砍，这指尖凝聚又消散的魔力微光，究竟是为了什么？

　　夕阳熔金，将村庄涂抹上一层暖而疲惫的橘红。晚餐的简单食物——黑麦面包、炖豆子和一点熏肉，带着烟火的气息，却无法驱散心头的滞涩。
　　
　　我沉默地咀嚼着，味同嚼蜡，视线低垂，盯着木碗边缘一道细小的裂纹。

　　“艾尔。”

　　父亲的声音不高，却像一块投入平静水面的石头，瞬间击碎了我的沉默。他放下木勺，那双历经风霜、仿佛能洞穿迷雾的眼睛平静地落在我身上，带着一种沉甸甸的审视。炉膛里跳跃的火焰在他深邃的眼眸里投下跳动的光影，让那目光显得更加锐利。

　　我抬起头，迎上他的视线，喉咙有些发紧，想说什么，却只觉一股难以言喻的委屈和迷茫堵在胸口。

　　“最近，你的剑，挥得心不在焉。像在砍一团没有形体的雾。”他顿了顿，粗糙的手指在木桌边缘无意识地摩挲了一下，“心里有事？”

　　平静的询问如同一把钥匙，猝然拧开了心底那扇紧闭的门。压抑了许久的情绪瞬间找到了出口，混杂着不甘、焦虑和深深的自责，一股脑地涌了上来，声音竟带上了一丝自己都未察觉的沙哑：
　　
　　“贝拉在救死扶伤，克罗埃在使用白魔法祛除病痛，欧斯的弓箭守护着村庄的边界，就连伊万，也去了锻火城邦追寻他的铁砧之道……”
　　
　　我的目光扫过空荡的桌面，仿佛伙伴们忙碌的身影就在眼前晃动。
　　
　　“而我呢？父亲大人？我每天只是在练习场上挥剑，在房间里对着空气凝聚魔力！像个……像个被圈养起来的废物！我的力量，我的练习，对村子，对大家，有什么用？”
　　
　　最后一句几乎是低吼出来，带着少年人特有的、被挫败感灼烧的尖锐。

　　父亲静静地听着，脸上没有责备，也没有急于安慰。他深邃的目光越过我的头顶，仿佛穿透了小屋简陋的墙壁，投向某个遥远而宏大的所在。
　　
　　炉火的光在他轮廓分明的脸上跳跃，那曾属于年轻勇者的放荡不羁早已沉淀，只剩下岁月和重担打磨出的坚硬棱角。
　　
　　“用处么……村子东头的老约翰，他那破风车吱嘎响得全村都头疼，缺个手脚利落的人去修修轴承。或者，贝拉的小药圃，听说最近野兔闹得凶，篱笆要加固。这小村子里力气活有的是，总能派上用场的。”
　　
　　他声音低沉，带着一种磨砺刀锋般的质感。
　　
　　我怔了一下。修风车？扎篱笆？这并非我预想中关于力量与意义的探讨。一股更深的无力感涌上来，夹杂着一丝被误解的委屈。
　　
　　“父亲大人，我不是怕出力，也不是嫌活计粗重……”我试图寻找更准确的词句，“我是说……我的剑术，我的魔法，它们……不该仅仅用来对付轴承和野兔吧？”
　　
　　父亲端起杯子的动作顿了一下，他抬眼，目光越过手中那杯麦茶，落在我脸上，那双曾凝视过魔王城深处黑暗的眼睛，此刻带着一点温和的、近乎促狭的了然。
　
　　“艾尔，不知不觉，你已经长大了啊……”父亲长长地叹了一口气，有些释然。
　　
　　“你说的没错，你的剑，你的魔法，不该仅仅困在这片屋檐之下。它们的光芒，注定要照亮更远的地方。”

　　他啜饮了一口麦茶，动作不疾不徐。
　　
　　“说起来，王都那边，明年暑期，魔法学院好像又要开始招生了，你或许可以去那里碰碰运气。”

　　“魔法……学院？”我的声音不自觉地放轻了，带着疑惑，同时也带着如同在迷雾中发现灯塔般的欣喜。

　　“嗯。”父亲应了一声，仿佛在确认一个再普通不过的消息，他拿起水壶，将空掉的杯子再次蓄满茶水。
　　
　　“魔法学院，毫无疑问的是王国中所有新生代真正的摇篮。那里汇聚着最全面的魔法典籍，最优质的导师，同时学校也注重全面发展，对剑术，体术等方面也均有培养。”

　　“那里……”我听见自己的声音有些发干，胸腔里有什么东西在有力地搏动，“我……能行？”
　　
　　“以你目前展现的魔法天赋，和你剑术打下的根基，通过那里暑期的入学考试，并非难事。”
　　
　　父亲轻轻地点了点头，肯定了我的实力。
　　
　　“魔法学院每届的优秀毕业生，将被冠以「神谕者」之名。而「神谕者」，拥有参与四年一度「星辉试炼」的唯一资格。”
　　
　　“星辉试炼？”
　　
　　“简单来说，就是选拔勇者队伍的考试。这是一条我当年曾经走过的路，并不轻松。我不会强迫你的选择，所有的决定一切在于你，你需要自己决定自己脚下的路。”
　　　
　　窗外的村庄早已沉入夜色，一片安宁。而我的目光，却仿佛穿透了这静谧的黑暗，投向了南方那片被无数传说和星辉笼罩的、充满无限可能的辽阔天地。
　　
　　指尖似乎还残留着魔力流动的微弱触感，此刻它不再虚无，仿佛第一次触碰到了某种坚实而具体的未来轮廓。\n\n
"""
    this_passage_request = """
\n**阶段三：书山有路勤为径**

1.  **认知的落差：严峻的试炼要求**
    *   目标明确后，艾尔立刻着手了解魔法学院的入学试炼详情。然而，获取的信息（通过父亲口述和简章）让他瞬间意识到巨大的挑战。**试炼的核心难点并非他擅长的领域：**
        *   **武技与魔法：** 基础剑术/格斗能力达标、元素亲和力、精神力稳定性、基础法术施展——这些对艾尔而言早已是熟练掌握甚至超出标准的能力，只需保持状态即可。
        *   **真正的壁垒：** **王国历史与地理沿革、基础政治架构（复杂的贵族谱系、议会职能、王权象征）、实用算术（测量、基础几何、物资分配计算）。** 这些对于成长在偏远村落的艾尔来说，是完全陌生的知识海洋！他感到一阵眩晕，仿佛要攀登一座从未见过的险峰。

2.  **孤独的攀登：自学与父教**
    *   面对这堵由书本知识筑成的高墙，艾尔别无选择，只能依靠自己（和有限的家庭资源）开始艰苦的攻坚。
    *   **历史与政治：** 成为艾尔最大的噩梦。卡伦提供的有限书籍（可能是一本陈旧的王国简史和一份模糊的领地地图）成了他唯一的依靠。他必须自己啃读那些晦涩难懂的文字，记忆拗口的国王名讳、战争年代、领地划分和错综复杂的贵族关系。常常在油灯下看得头昏眼花，进展缓慢而痛苦。卡伦能提供的帮助有限，只能在他遇到完全无法理解的关键节点时，进行最基础的澄清。
    *   **算术：** 这部分则由父亲卡伦亲自教导。卡伦结合村庄的实际需求及村庄事务（土地丈量、收成分配、日期推算），教导艾尔更实用的算术知识。从加减乘除的巩固，到比例、面积、体积的计算。艾尔虽然对数字依然不算敏感，但在卡伦结合实例的耐心讲解下，理解比枯燥的书本要好得多，至少能掌握基础方法。不过，复杂的题目仍让他皱眉不已。
    *   **剑术与魔法：** 这两项超标的强项，此刻成为了艾尔繁重学习之余难得的放松和“避难所”。在练习场上挥洒汗水，或是沉浸于魔力的流动中，能让他暂时逃离那些令人头疼的公式和历史事件的泥沼。卡伦也默契地减少了这方面的指导时间，仅维持其状态，将主要精力留给文化课攻坚。

3.  **伙伴的轨迹与无声的鞭策**
    *   在艾尔埋头苦读时，伙伴们各自的生活仍在向前：
        *   **贝拉与克罗埃：** 继续承担着村中主要的医疗工作，日益熟练。
        *   **欧斯：** 作为村卫队成员，巡逻守卫是日常职责。
        *   **伊万：** 远在矮人国，杳无音信。
    *   他们各自忙碌，无暇像过去那样频繁相聚。艾尔偶尔从窗户看到贝拉匆匆背着药篓走过，或欧斯全副武装地出发巡逻，心中会掠过一丝“大家都在为村庄做事”的念头。这并非强烈的压力或对比，更像是一种淡淡的背景提示：**世界在运转，每个人都有自己的位置和责任，而他选择的道路，需要独自跨越眼前的障碍。** 伙伴们的高歌猛进不再是直接的压力源，而是转化为一种“我也必须前进”的自我鞭策。

4.  **克罗埃的抉择**
    *   当艾尔深陷书海之际，一个关于克罗埃的重要选择摆在面前。邻近的**修道院**院长听闻克罗埃在治疗魔法上的天赋，尤其是其温和纯净的白魔法，发来信函，诚挚邀请克罗埃前往修道院接受系统、深入的治疗魔法与神圣知识教育，并提供珍贵的学徒名额。
    *   这对克罗埃是难得的机遇，意味着更专业的培养和更广阔的未来。然而，他也深知自己是村庄重要的医疗力量，最终放弃了机会，将自己奉献给这片在收留自己的村落。\n\n"""
    article_name = "入学篇"
    chapter_number = "64"
    chapter_name = "知识补习"
    characters = ["艾尔维/艾尔","贝拉","欧斯","伊万","克罗埃","卡伦"]
    world = tools.read_txt_file(os.path.join(NOVEL_WORLD_FOLDER, "常用世界观.txt"))
    styles = tools.read_txt_file(os.path.join(NOVEL_STYLES_FOLDER, "常用文风.txt"))
    words = 6000
    is_NPC = False

    print(writting_novel(last_passage,
                        this_passage_request,
                        article_name,
                        chapter_number, 
                        chapter_name,
                        characters,
                        world,
                        styles,
                        words,
                        is_NPC))

    
    