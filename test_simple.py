import torch
import torchtext
import torchsnooper
from torchtext.data import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#构建词典
train_iter = IMDB(split='train')#25000条评论
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
# This index will be returned when OOV(out of vacabulary) token is queried.

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_pipeline = lambda x:int(0) if x == 'neg' else int(1)
text_pipeline = lambda x:vocab(tokenizer(x))

def collate_fn(batch):
    label_list, text_list = [],[]
    # batch.sort(key= lambda x:len(x[0]))
    for (label,text) in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        len_text = len(processed_text) #为了将tensor第一维度小于20的与0拼接
        # 这里的条件语句是为了使得每个句子长度为20
        if len(processed_text) >=20:
            text_list.append(processed_text[:20])
        else:
            catted_tensor = torch.zeros((20-len_text),dtype=torch.int64)
            cat_tensor = torch.cat((processed_text,catted_tensor),0)
            text_list.append(cat_tensor)

    label_list = torch.tensor(label_list, dtype=torch.float32)
    text_list = torch.cat(text_list,dim=0)
    return label_list.to(device), text_list.to(device)

# 包含tensor的list转tensor：list(tensors) → torch.cat(list, dim=0)
#
# 包含整数或浮点数的list转tensor：list(int/float) → torch.tensor(list)

train_iter = IMDB(split='train')
train_dataloader = DataLoader(train_iter, batch_size=32, shuffle=False,collate_fn=collate_fn,drop_last=True)
# train_dataloader = DataLoader(train_iter, batch_size=1, shuffle=False,collate_fn=collate_fn)实验用的
test_iter = IMDB(split='test')
test_dataloader = DataLoader(test_iter, batch_size=32, shuffle=False,collate_fn=collate_fn,drop_last=True)



# ls = []
# for label,text in train_iter:
#     processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
#     len_text = len(processed_text)
#     if len(processed_text) >=20:
#         ls.append(processed_text[:20])
#     else:
#         catted_tensor = torch.zeros((20-len_text))
#         cat_tensor = torch.cat((processed_text,catted_tensor),0)
#         ls.append(cat_tensor)
#
# print(ls)

# a = torch.Tensor([1,2,3,4])
# b = torch.zeros(3)
# c = torch.cat((a,b),0)
# print(c)


from torch import nn

class LogisticRegressionModel(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(LogisticRegressionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.flatten = nn.Flatten()#为啥slide这里要打平?
        self.fc = nn.Linear(32*20*embed_dim,32,bias=True)#fc means fully connected layer
        self.sigmoid = nn.Sigmoid()


    def forward(self, text):
        embedded = self.embedding(text)
        flatted = embedded.flatten()
        out = self.fc(flatted)
        out = F.sigmoid(out)

        return out

train_iter = IMDB(split='train')#又声明一次，为啥啊
vocab_size = len(vocab)
embed_dim = 8 #王树森说8是cross validation选出来的，目前没明白
model = LogisticRegressionModel(vocab_size, embed_dim)
model.to(device)





# for batch_idx,(label,text) in enumerate(train_dataloader):
#     print(model(text).shape)

# '''
# 3124
# [('pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos'),
# ("This movie really kicked some ass. I watched it over and over and it never got boring. Angelina Jolie really kicked some ass in the movie, you should see the movie, you won't be disappointed. And another reason you should see the movie is because the guy from The X-Files is in it, David Duchovny.",
# "With the mixed reviews this got I wasn't expecting too much, and was pleasantly surprised. It's a very entertaining small crime film with interesting characters, excellent portrayals, writing that's breezy without being glib, and a good pace. It looks good too, in a funky way. Apparently people either like this movie or just hate it, and I'm one who liked it.",
# "Very smart, sometimes shocking, I just love it. It shoved one more side of David's brilliant talent. He impressed me greatly! David is the best. The movie captivates your attention for every second.", "A hit at the time but now better categorised as an Australian cult film. The humour is broad, unsubtle and, in the final scene where a BBC studio fire is extinguished by urinating on it, crude. Contains just about every cliche about the traditional Australian pilgrimage to 'the old country', and every cliche about those rapacious, stuck up, whinging, Tory Brits. Would be acceptable to the British because of its strong cast of well known actors, and to Australians of that generation, who can 'get' the humour. Americans -- forget it. The language and jokes are in the Australian dialect of English and as such will be unintelligible.",
# 'I love this movie like no other. Another time I will try to explain its virtues to the uninitiated, but for the moment let me quote a few of pieces the remarkable dialogue, which, please remember, is all tongue in cheek. Aussies and Poms will understand, everyone else-well?<br /><br />(title song lyric)"he can sink a beer, he can pick a queer, in his latest double-breasted Bondi gear."<br /><br />(another song lyric) "All pommies are bastards, bastards, or worse, and England is the a**e-hole of the universe."<br /><br />(during a television interview on an "arty program"): Mr Mackenzie what artists have impressed you most since you\'ve been in England? (Barry\'s response)Flamin\' bull-artists!<br /><br />(while chatting up a naive young pom girl): Mr Mackenzie, I suppose you have hordes of Aboriginal servants back in Australia? (Barry\'s response) Abos? I\'ve never seen an Abo in me life. Mum does most of the solid yacca (ie hard work) round our place.<br /><br />This is just a taste of the hilarious farce of this bonser Aussie flick. If you can get a copy of it, watch and enjoy.',
# "This film and it's sequel Barry Mckenzie holds his own, are the two greatest comedies to ever be produced. A great story a young Aussie bloke travels to england to claim his inheritance and meets up with his mates, who are just as loveable and innocent as he is.<br /><br />It's chock a block full of great, sayings , where else could you find someone who needs a drink so bad that he's as dry as a dead dingoes donger? great characters, top acting, and it's got great sheilas and more Fosters consumption then any other three films put together. Top notch.<br /><br />And some of the funniest songs you'll ever hear, and it's full of great celebrities. Definitely my two favourite films of all time, I watch them at least once a fortnight.", '\'The Adventures Of Barry McKenzie\' started life as a satirical comic strip in \'Private Eye\', written by Barry Humphries and based on an idea by Peter Cook. McKenzie ( \'Bazza\' to his friends ) is a lanky, loud, hat-wearing Australian whose two main interests in life are sex ( despite never having had any ) and Fosters lager. In 1972, he found his way to the big screen for the first of two outings. It must have been tempting for Humphries to cast himself as \'Bazza\', but he wisely left the job to Barry Crocker ( later to sing the theme to the television soap opera \'Neighbours\'! ). Humphries instead played multiple roles in true Peter Sellers fashion, most notably Bazza\'s overbearing Aunt \'Edna Everage\' ( this was before she became a Dame ).<br /><br />You know this is not going to be \'The Importance Of Being Ernest\' when its censorship classification N.P.A. stands for \'No Poofters Allowed\'. Pom-hating Bazza is told by a Sydney solicitor that in order to inherit a share in his father\'s will he must go to England to absorb British culture. With Aunt Edna in tow, he catches a Quantas flight to Hong Kong, and then on to London. An over-efficient customs officer makes Bazza pay import duties on everything he bought over there, including a suitcase full of \'tubes of Fosters lager\'. As he puts it: "when it comes to fleecing you, the Poms have got the edge on the gyppos!". A crafty taxi driver ( Bernard Spear ) maximises the fare by taking Bazza and Edna first to Stonehenge, then Scotland. The streets of London are filthy, and their hotel is a hovel run by a seedy landlord ( Spike Milligan ) who makes Bazza put pound notes in the electricity meter every twenty minutes. There is some good news for our hero though; he meets up with other Aussies in Earls Court, and Fosters is on sale in British pubs.<br /><br />What happens next is a series of comical escapades that take Bazza from starring in his own cigarette commercial, putting curry down his pants in the belief it is some form of aphrodisiac, a bizarre encounter with Dennis Price as an upper-class pervert who loves being spanked while wearing a schoolboy\'s uniform, a Young Conservative dance in Rickmansworth to a charity rock concert where his song about \'chundering\' ( vomiting ) almost makes him an international star, and finally to the B.B.C. T.V. Centre where he pulls his pants down on a live talk-show hosted by the thinking man\'s crumpet herself, Joan Bakewell. A fire breaks out, and Bazza\'s friends come to the rescue - downing cans of Fosters, they urinate on the flames en masse.<br /><br />This is a far cry from Bruce Beresford\'s later works - \'Breaker Morant\' and \'Driving Miss Daisy\'. On release, it was savaged by critics for being too \'vulgar\'. Well, yes, it is, but it is also great non-P.C. fun. \'Bazza\' is a disgusting creation, but his zest for life is unmistakable, you cannot help but like the guy. His various euphemisms for urinating ( \'point Percy at the porcelain\' ) and vomiting ( \'the Technicolour yawn\' ) have passed into the English language without a lot of people knowing where they came from. Other guest stars include Dick Bentley ( as a detective who chases Bazza everywhere ), Peter Cook, Julie Covington ( later to star in \'Rock Follies\' ), and even future arts presenter Russell Davies.<br /><br />A sequel - the wonderfully-named \'Barry McKenzie Holds His Own - came out two years later. At its premiere, Humphries took the opportunity to blast the critics who had savaged the first film. Good for him.<br /><br />What must have been of greater concern to him, though, was the release of \'Crocodile Dundee\' in 1985. It also featured a lanky, hat-wearing Aussie struggling to come to terms with a foreign culture. And made tonnes more money.<br /><br />The song on the end credits ( performed by Snacka Fitzgibbon ) is magnificent. You have a love a lyric that includes the line: "If you want to send your sister in a frenzy, introduce her to Barry McKenzie!". Time to end this review. I have to go the dunny to shake hands with the unemployed...', '
# The story centers around Barry McKenzie who must go to England if he wishes to claim his inheritance. Being about the grossest Aussie shearer ever to set foot o
# utside this great Nation of ours there is something of a culture clash and much fun and games ensue.
# The songs of Barry McKenzie(Barry Crocker) are highlights.')]
#
# 这个是增加collate_fn之后的函数
# 3124
# (tensor([1, 1, 1, 1, 1, 1, 1, 1]),
# tensor([  13,   20,   71,  ...,   30, 3538,    2]))
# '''
# # for batch_idx,(label,text) in enumerate(train_dataloader):
# #     print(model(text).shape)
# # torch.Size([1805, 8])
# # torch.Size([2298, 8])
# # torch.Size([2387, 8])
#


LR = 0.0001
optimzier = torch.optim.SGD(model.parameters(), lr=LR)

def train_loop(dataloader,model,optimizer):
    model.train()
    for batch_idx,(label,text) in enumerate(dataloader):
        optimizer.zero_grad()#set gradient to zero
        predicted_label = model(text)
        # print(predicted_label)
        loss = F.binary_cross_entropy(predicted_label,label)#了解一下这个损失函数与crossentropy
        loss.backward()#compute gradient
        optimzier.step()#unpdate model with optimizer
        print('Training Loss :{}'.format(loss))





def test_loop(dataloader,model):
    model.eval()
    totalacc,acc_count= 0,0
    with torch.no_grad():#disable gradient calculation
        for batch_idx,(label,text) in enumerate(dataloader):
            pred = model(text)
            loss = F.binary_cross_entropy(pred,label)
            rounded_pred = torch.round(loss)#四舍五入
            correct = (rounded_pred == label).float()
            acc = correct.sum() / len(correct)
            # acc_count += 1
            # totalacc += acc
            print('Accuracy:{}'.format(acc))


epochs = 10

#记录错误，没有输出每个iteraion的loss和acc，而是把每次train的值都打印出来了，弄混了epoch，iteration以及batcg_size的关系
#输出了781个acc？
for t in range(epochs):
    print("Epoch {}\n-------------------------------".format(t+1))
    train_loop(train_dataloader, model, optimzier)
    test_loop(test_dataloader, model)
print("Done!")

