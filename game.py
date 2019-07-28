import pygame, sys, time, random
from pygame.locals import *
import numpy as np
import matrix_define as M
import threading
# import game_train as gt
# import game_eval as ge
import tensorflow as tf
import os
import fcn

FPS = 30
WIDTH = 400
HEIGHT = 480
BOXSIZE = 20
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
SPEED = 0.00001
height_weight = 2
blank_weight = 1
MAX_SCORE = 0
FONT_PATH = 'font/SimHei.ttf'

WHITE = (255, 255, 255)
GRAY = (185, 185, 185)
BLACK = (189, 167, 0)
RED = (155, 0, 0)
GREEN = (0, 155, 0)
BLUE = (0, 191, 255)
YELLOW = (155, 155, 0)


BATCH_SIZE = 16

LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.9
REGULA_RATE = 0.0001
TRAING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99


MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


Color = (WHITE, GRAY, BLACK, RED, GREEN, BLUE, YELLOW)
DEMO_MATRIX = M.init_demo_matrix()
board_root = [30, 30]
demomat_root = [260, 50]
LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DATA_FLAG = 0
data_speed = 0.01
train_speed = 0.1
MODEL = 3
'''
0 - 玩家模式
1 - 手动训练
2 - 全自动训练
3 - ai模式
'''


def main():
    global screen, FPSCLOCK, dataset, data, score
    score = []
    data = []
    dataset = []
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill(BLUE)
    pygame.display.set_caption('炫彩俄罗斯')
    pygame.draw.rect(screen, WHITE, (board_root[0], board_root[1], BOARD_WIDTH * BOXSIZE, BOARD_HEIGHT * BOXSIZE), 0)
    pygame.draw.rect(screen, WHITE, (demomat_root[0], demomat_root[1], BOXSIZE * 4, BOXSIZE * 4), 0)
    # drawText(text="hellowworld", posx= 1, posy=1, fontColor=WHITE)
    # if MODEL != 0:
    # t = threading.Thread(target=trainx)
    # t.setDaemon(True)
    # t.start()
    # d = threading.Thread(target=data_process)
    # d.setDaemon(True)
    # d.start()

    # autogame()
    rungame()


def matrix_prosessx(matrix, i1, j1, x, y):
    # print('matrix_prosess')
    h1 = matrix_height(matrix)
    b1 = matrix_blank(matrix)
    matrix = add_shape(matrix, i1, j1, x, y)
    matrix = matrix_del(matrix)
    h2 = matrix_height(matrix)
    b2 = matrix_blank(matrix)
    height_change = h2 - h1
    blank_change = b2 - b1
    # data.append([matrix, i1, j1, x, height_change, blank_change])
    return matrix, h2


def calculate(matrix, shape, j, x):
    h1 = matrix_height(matrix)
    b1 = matrix_blank(matrix)

    y = 0
    try:
        if not check_board(matrix, shape[j], x, y):
            return 999
    except:
        return 999
    while check_fall(matrix, shape[j], x, y+1):
        y += 1
    # print(shape[j], x, y)
    matrix = add_shape(matrix, shape, j, x, y)
    matrix = matrix_del(matrix, False)
    h2 = matrix_height(matrix)
    b2 = matrix_blank(matrix)
    variance = matrix_varience(matrix)
    height_change = h2 - h1
    blank_change = b2 - b1
    if height_change >= 0:
        loss = height_change * 1 + blank_change * 6 + variance * 5
    else:
        loss = height_change * 4 + blank_change * 6 + variance * 2
    # print('shape:')
    # print(np.array(shape[j]))
    # print("x:%d, loss:%d" % (x, loss))
    return loss


def matrix_varience(matrix):
    a = np.array(matrix)
    b = a.sum(axis=0)
    # print()
    # print(b)
    result = b.var()
    return result


def evalue(matrix, shape):
    # print('evalue')
    min_x = random.randint(0, 9)
    min_j = random.randint(0, len(shape))
    min_result = calculate(matrix, shape, min_j, min_x)
    # print('init shape:')
    # print(np.array(shape[min_j]))
    # print("x:%d, loss:%d" % (min_x, min_result))
    for m in range(10):
        for n in range(len(shape)):
            current_result = calculate(matrix, shape, n, m)
            if current_result < min_result:
                min_result = current_result
                min_x = m
                min_j = n
    return min_x, min_j


def rungame():
    # if MODEL == 2 or MODEL == 3:
    print('rungame')
    # score = []

    for r in range(10):
        matrix = init_matrix()
        play = True
        print("new play,round: %d", r)
        i1, j1, current_shape = get_demomat()
        while play:
            # print("new fall")

            i2, j2, next_shape = get_demomat()
            x = 4
            y = 0
            draw_nextmat(next_shape[j2])
            press_down = False
            current_fall = True
            draw_mat(current_shape[j1], x, y)

            min_x, min_j = evalue(matrix, current_shape)
            # print(min_x, min_j)
            min_j = min_j % len(current_shape)
            timer = time.time()
            while current_fall:
                # print("new fall")
                for event in pygame.event.get():
                    if event.type == KEYUP:
                        if (event.key == K_DOWN):
                            press_down = False
                    if event.type == KEYDOWN:
                        if (event.key == K_LEFT) and check_board(matrix, current_shape[j1], x-1, y):
                            x -= 1

                        elif (event.key == K_RIGHT) and check_board(matrix, current_shape[j1], x+1, y):
                            x += 1

                        elif (event.key == K_UP):
                            j_ = (j1 + 1) % len(current_shape)
                            if check_board(matrix, current_shape[j_], x, y):
                                j1 = j_

                        elif (event.key == K_DOWN):
                            press_down = True
                            if check_fall(matrix, current_shape[j1], x, y + 1):
                                    y += 1
                            else:
                                current_fall = False

                    if event.type == QUIT:
                        exit()

                if j1 < min_j:
                    j_ = (j1 + 1) % len(current_shape)
                    if check_board(matrix, current_shape[j_], x, y):
                        j1 = j_
                if j1 > min_j:
                    j1 -= 1

                if x < min_x:
                    if check_board(matrix, current_shape[j1], x + 1, y):
                        x += 1
                if x > min_x:
                    x -= 1

                if press_down and time.time() - timer > 0.1:
                    if check_fall(matrix, current_shape[j1], x, y + 1):
                        y += 1
                    else:
                        current_fall = False

                if time.time() - timer > SPEED:
                    if not check_fall(matrix, current_shape[j1], x, y + 1):
                        current_fall = False
                    else:
                        y += 1
                    timer = time.time()
                    # print('1111')

                draw_content(matrix)
                draw_mat(current_shape[j1], x, y)
                if not current_fall:
                    matrix, h = matrix_prosess(matrix, i1, j1, x, y, True)
                    # print('11111111111111111111')
                    print('varience:', matrix_varience(matrix))
                    i1, j1, current_shape = i2, j2, next_shape
                    if h == 20 or not check_fall(matrix, next_shape[j2], 4, 0):
                        play = False

                pygame.display.update()
                FPSCLOCK.tick(FPS)


def autogame():
    # if MODEL == 2 or MODEL == 3:

    print('rungame')
    while True:
        matrix = init_matrix()
        play = True
        print("new play")
        i1, j1, current_shape = get_demomat()
        while play:
            print("new fall")

            i2, j2, next_shape = get_demomat()
            x = random.randint(0, 6)
            y = 0
            draw_nextmat(next_shape[j2])
            press_down = False
            current_fall = True

            with tf.Graph().as_default() as g:
                x_ = tf.placeholder(tf.float32, [None, fcn.INPUT_NODE], name='x-input')
                y_ = tf.placeholder(tf.float32, [None, fcn.OUTPUT_NODE], name='y-input')
                result = fcn.interface(x_, None)

                variable_averages = tf.train.ExponentialMovingAverage(
                    MOVING_AVERAGE_DECAY)
                variable_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variable_to_restore)
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        print('current model', ckpt.model_checkpoint_path)
                        saver.restore(sess, ckpt.model_checkpoint_path)

                        min_i = random.randint(0, 9)
                        min_j = random.randint(0, 3)

                        mat = matrix
                        shape_list = index_list(7, j1)
                        tag_list = index_list(4, min_j)
                        x_list = index_list(10, min_i)
                        input_list = [mat, shape_list, tag_list, x_list]
                        s = str(input_list)
                        s = s.replace('[', '')
                        s = s.replace(']', '')
                        input_list = list(eval(s))
                        outputlist = [0.1, 0.1]
                        demo = [input_list, outputlist]
                        min_result = sess.run(result, feed_dict={x_: [demo[0]], y_: [demo[1]]})

                        for i in range(10):
                            for j in range(4):
                                mat = matrix
                                shape_list = index_list(7, j1)
                                tag_list = index_list(4, j)
                                x_list = index_list(10, i)
                                input_list = [mat, shape_list, tag_list, x_list]
                                s = str(input_list)
                                s = s.replace('[', '')
                                s = s.replace(']', '')
                                input_list = list(eval(s))
                                outputlist = [0.1, 0.1]
                                demo = [input_list, outputlist]
                                # print(input_list)
                                # print(outputlist)
                                # output_list = [d[4], d[5]]
                                current_result = sess.run(result, feed_dict={x_: [demo[0]], y_: [demo[1]]})
                                if sum(current_result[0]) < sum(min_result[0]):
                                    min_i = i
                                    min_j = j
                        min_j = min_j % len(current_shape)
                        # game_eval(matrix, i1, j1, x)

            timer = time.time()
            while current_fall:
                # print("new fall")
                for event in pygame.event.get():
                    if event.type == KEYUP:
                        if (event.key == K_DOWN):
                            press_down = False
                    if event.type == KEYDOWN:
                        if (event.key == K_LEFT) and check_board(matrix, current_shape[j1], x - 1, y):
                            x -= 1

                        elif (event.key == K_RIGHT) and check_board(matrix, current_shape[j1], x + 1, y):
                            x += 1

                        elif (event.key == K_UP):
                            j_ = (j1 + 1) % len(current_shape)
                            if check_board(matrix, current_shape[j_], x, y):
                                j1 = j_

                        elif (event.key == K_DOWN):
                            press_down = True
                            if check_fall(matrix, current_shape[j1], x, y + 1):
                                y += 1
                            else:
                                current_fall = False

                    if event.type == QUIT:
                        exit()

                if press_down and time.time() - timer > 0.1:
                    if check_fall(matrix, current_shape[j1], x, y + 1):
                        y += 1
                    else:
                        current_fall = False

                if time.time() - timer > SPEED:
                    if not check_fall(matrix, current_shape[j1], x, y + 1):
                        current_fall = False
                    else:
                        y += 1
                    timer = time.time()
                    # print('1111')

                if j1 < min_j:
                    j_ = (j1 + 1) % len(current_shape)
                    if check_board(matrix, current_shape[j_], x, y):
                        j1 = j_
                if j1 > min_j:
                    j1 -= 1

                if x > min_i:
                    x -= 1
                if x < min_j:
                    x += 1

                draw_content(matrix)
                draw_mat(current_shape[j1], x, y)
                if not current_fall:
                    matrix, h = matrix_prosess(matrix, i1, j1, x, y)

                    i1, j1, current_shape = i2, j2, next_shape
                    if h == 20 or not check_fall(matrix, next_shape[j2], 4, 0):
                        play = False

                pygame.display.update()
                FPSCLOCK.tick(FPS)


def trainx():
    # print('load train config')
    x = tf.placeholder(tf.float32, [None, fcn.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, fcn.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULA_RATE)

    y = fcn.interface(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cost_function = tf.reduce_mean(tf.square(y_ - y))
    # cost = cost_fun(x, y)
    loss = cost_function + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, 20, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver(max_to_keep=1)
    print('train config load ')

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            new_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            if int(new_step) >= TRAING_STEP:
                while True:
                    print('train times reach the max')
            current_step = int(new_step)
            print('have train %d step' % current_step)
            i = 0
            # for i in range(int(new_step), gt.TRAING_STEP):
            # print(len(dataset))
            while current_step < TRAING_STEP:
                if int(len(dataset) / BATCH_SIZE) > i:
                    xs, ys = get_batch(dataset[i * BATCH_SIZE:(i+1) * BATCH_SIZE])
                    # print('test111111111')
                    # ge.eval_config.eval(xs, ys)

                    _, loss_value, step, pre_y = \
                        sess.run([train_op, loss, global_step, y], feed_dict={x: xs, y_: ys})
                    print(pre_y)
                    print("After %d training step(s), loss on training batch is %g.predition y is :" % (step, loss_value))
                    # print(y)
                    i += 1
                    current_step += 1
                    if current_step % 10 == 0:

                        print("After %d training step(s), loss on training batch is %g. save mode" % (step, loss_value))
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print("waitting for dataset num: %d, have train %d step this time" % (len(dataset), i))
                time.sleep(train_speed)
        else:
            tf.global_variables_initializer().run()
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            # tf.global_variables_initializer().run()
            #
            # for i in range(TRAING_STEP):
            #
            # xs, ys = tr.get_batch(dataset, gt.BATCH_SIZE)
            # _, loss_value, step = \
            #     tr.sess.run([tr.train_op, tr.loss, tr.global_step], feed_dict={tr.x: xs, tr.y_: ys})
            #
            # print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            # tr.saver.save(sess, os.path.join(gt.MODEL_SAVE_PATH, gt.MODEL_NAME), global_step=tr.global_step)

            # if step % 10 == 0:
            #
            #     print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            #     tr.saver.save(sess, os.path.join(gt.MODEL_SAVE_PATH, gt.MODEL_NAME), global_step=tr.global_step)


def get_batch(dataset):
    xs = []
    ys = []
    for i in range(BATCH_SIZE):
        xs.append(dataset[i][0])
        ys.append(dataset[i][1])
    print(ys)
    return xs, ys


# def train():
#     # # print('load train config')
#     # x = tf.placeholder(tf.float32, [None, fcn.INPUT_NODE], name='x-input')
#     # y_ = tf.placeholder(tf.float32, [None, fcn.OUTPUT_NODE], name='y-input')
#     #
#     # regularizer = tf.contrib.layers.l2_regularizer(REGULA_RATE)
#     #
#     # y = fcn.interface(x, regularizer)
#     # global_step = tf.Variable(0, trainable=False)
#     #
#     # variable_averages = tf.train.ExponentialMovingAverage(
#     #     MOVING_AVERAGE_DECAY, global_step)
#     # variable_averages_op = variable_averages.apply(tf.trainable_variables())
#     #
#     # cost_function = tf.reduce_mean(tf.square(y_ - y))
#     # # cost = cost_fun(x, y)
#     # loss = cost_function + tf.add_n(tf.get_collection('losses'))
#     # learning_rate = tf.train.exponential_decay(
#     #     LEARNING_RATE_BASE, global_step, 50, LEARNING_RATE_DECAY)
#     # train_step = tf.train.GradientDescentOptimizer(learning_rate) \
#     #     .minimize(loss, global_step=global_step)
#     # with tf.control_dependencies([train_step, variable_averages_op]):
#     #     train_op = tf.no_op(name='train')
#     # saver = tf.train.Saver(max_to_keep=1)
#     # # self.sess = tf.Session()
#     print('train config load ')
#     tr = gt.train_config()
#
#     with tf.Session() as sess:
#
#         ckpt = tf.train.get_checkpoint_state(gt.MODEL_SAVE_PATH)
#         if ckpt and ckpt.model_checkpoint_path:
#             tr.saver.restore(sess, ckpt.model_checkpoint_path)
#             new_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#             if int(new_step) >= gt.TRAING_STEP:
#                 while True:
#                     print('train times reach the max')
#             current_step = int(new_step)
#             print('have train %d step' % current_step)
#             i = 0
#             # for i in range(int(new_step), gt.TRAING_STEP):
#             # print(len(dataset))
#             while current_step < gt.TRAING_STEP:
#                 if int(len(dataset) / gt.BATCH_SIZE) > i:
#                     xs, ys = tr.get_batch(dataset[i * gt.BATCH_SIZE:(i+1) * gt.BATCH_SIZE])
#                     # print('test111111111')
#                     # ge.eval_config.eval(xs, ys)
#
#                     _, loss_value, step, y = \
#                         sess.run([tr.train_op, tr.loss, tr.global_step, tr.y], feed_dict={tr.x: xs, tr.y_: ys})
#                     print("After %d training step(s), loss on training batch is %g.predition y is :" % (step, loss_value))
#                     print(y)
#                     i += 1
#                     current_step += 1
#                     if current_step % 10 == 0:
#
#                         print("After %d training step(s), loss on training batch is %g. save mode" % (step, loss_value))
#                         tr.saver.save(sess, os.path.join(gt.MODEL_SAVE_PATH, gt.MODEL_NAME),
#                                       global_step=tr.global_step)
#                 print("waitting for dataset num: %d, have train %d step this time" % (len(dataset), i))
#                 time.sleep(train_speed)
#         else:
#             tf.global_variables_initializer().run()
#             tr.saver.save(sess, os.path.join(gt.MODEL_SAVE_PATH, gt.MODEL_NAME), global_step=tr.global_step)
#
#             # tf.global_variables_initializer().run()
#             #
#             # for i in range(TRAING_STEP):
#             #
#             # xs, ys = tr.get_batch(dataset, gt.BATCH_SIZE)
#             # _, loss_value, step = \
#             #     tr.sess.run([tr.train_op, tr.loss, tr.global_step], feed_dict={tr.x: xs, tr.y_: ys})
#             #
#             # print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#             # tr.saver.save(sess, os.path.join(gt.MODEL_SAVE_PATH, gt.MODEL_NAME), global_step=tr.global_step)
#
#             # if step % 10 == 0:
#             #
#             #     print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#             #     tr.saver.save(sess, os.path.join(gt.MODEL_SAVE_PATH, gt.MODEL_NAME), global_step=tr.global_step)


def data_process():
    print('data_process--ing')
    while True:
        # isprocess = True
        # print('data num:', len(data))
        for d in data:
            mat = d[0]
            shape_list = index_list(7, int(d[1]))
            tag_list = index_list(4, int(d[2]))
            x_list = index_list(10, int(d[3]))
            input_list = [mat, shape_list, tag_list, x_list]
            s = str(input_list)
            s = s.replace('[', '')
            s = s.replace(']', '')
            input_list = list(eval(s))

            output_list = [d[4], d[5]]

            # ev = ge.eval_config()
            # demo = [input_list, output_list]
            # ev.eval(demo)

            # input_list = mat_list.extend(shape_list)
            # input_list = input_list.extend(tag_list)
            # input_list = input_list.extend(x_list)

            dataset.append([input_list, output_list])

            # print('add one datase:', shape_list, x_list)
            data.remove(d)

        # isprocess= False
        time.sleep(data_speed)


def index_list(num, index):
    li = []
    for i in range(num):
        if i == index:
            li.append(1)
        else:
            li.append(0)
    return li


def matrix_prosess(matrix, i1, j1, x, y, flag):
    # print('matrix_prosess')
    h1 = matrix_height(matrix)
    b1 = matrix_blank(matrix)
    matrix = add_shape(matrix, DEMO_MATRIX[i1], j1, x, y)
    matrix = matrix_del(matrix, flag)
    h2 = matrix_height(matrix)
    b2 = matrix_blank(matrix)
    height_change = h2 - h1
    blank_change = b2 - b1
    print('height:', height_change)
    print('blank:', blank_change)
    # data.append([matrix, i1, j1, x, height_change, blank_change])
    return matrix, h2


def matrix_blank(matrix):
    mat_arr = np.array(matrix)
    mat_trans = mat_arr.transpose()
    blank = 0
    for i in range(10):
        if sum(mat_trans[i]) != 0:
            for j in range(20):
                if mat_trans[i][j] != 0:
                    break
            blank += 20 - j - sum(mat_trans[i])
    return blank


def matrix_del(matrix, flag):
    # print('delet line')
    del_index = []
    for i in range(20):
        if sum(matrix[i]) > 9:
            del_index.append(i)
    if len(del_index) > 0:
        pass
        # print('del_index', del_index)
    for i in del_index:
        matrix.pop(i)
        matrix.insert(0, LINE)
        if flag:
            score.append(1)
    return matrix


def matrix_height(matrix):
    h = 0
    for i in range(20):
        if sum(matrix[i]) != 0:
            h += 1
    return h


def add_shape(matrix, shape, j1, x, y):
    # print('add_shape')
    mat = shape[j1]
    matrix = np.array(matrix)
    for m in range(4):
        for n in range(4):
            if mat[m][n] != 0:
                # print('add_point %d, %d'% (y+m, x+n))
                matrix[y + m][x + n] = 1
    matrix = matrix.tolist()
    return matrix


def check_fall(matrix, shape, x, y):
    flag = True
    for i in range(3, -1, -1):
        if (sum(shape[i]) > 0):
            for j in range(4):
                if shape[i][j] != 0:
                    if y + i > 19 or x + j > 9:
                        flag = False
                        break
                    if matrix[y+i][x+j] != 0:
                        flag = False
                        break
    return flag


def check_board(matrix, shape, x, y):
    flag = True
    if x < 0:
        return False
    for i in range(3, -1, -1):
        if (sum(shape[i]) > 0):
            for j in range(4):
                if shape[i][j] != 0:
                    if x + j > 9 or y + i > 19:
                        flag = False
                        break
                    if matrix[y+i][x+j] != 0:
                        flag = False
                        break
    return flag


def draw_mat(mat, x, y):
    c = random.randint(2, 6)
    for i in range(4):
        for j in range(4):
            if x+j < 10 and y+i < 20 and mat[i][j] != 0:
                pygame.draw.rect(screen, Color[c],
                            (board_root[0] + x * BOXSIZE + BOXSIZE * j, board_root[1] + y * BOXSIZE + BOXSIZE * i, BOXSIZE - 1, BOXSIZE - 1))


def get_demomat():
    i = random.randint(0, 6)
    j = random.randint(0, 3)
    j = j % len(DEMO_MATRIX[i])
    return i, j, DEMO_MATRIX[i]


def draw_nextmat(demo_mat):
    c = random.randint(2, 6)
    for i in range(4):
        for j in range(4):
            if demo_mat[i][j] != 0:
                demo_mat[i][j] = c
            pygame.draw.rect(screen, Color[demo_mat[i][j]],
                             (demomat_root[0] + BOXSIZE * j, demomat_root[1] + BOXSIZE * i, BOXSIZE-1, BOXSIZE-1))


def init_matrix():
    matrix = []
    for _ in range(20):
        line = []
        for _ in range(10):
            # line.append(random.randint(0, 6))
            line.append(0)
        matrix.append(line)
    return matrix


def draw_content(matrix):
    # print('draw_content')
    for i in range(20):
        for j in range(10):
            pygame.draw.rect(screen, Color[matrix[i][j]],
                             (board_root[0] + BOXSIZE * j, board_root[1] + BOXSIZE * i, BOXSIZE-1, BOXSIZE-1))


def drawText(text, posx=250, posy=350, textHeight=20, fontColor=RED, backgroudColor=GREEN):
    print('zdadawdaw')
    fontObj = pygame.font.Font(FONT_PATH, textHeight)  # 通过字体文件获得字体对象
    text_img = fontObj.render(text, True, fontColor, backgroudColor)  # 配置要显示的文字

    screen.blit(text_img, (posx, posy))  # 绘制字


if __name__ == '__main__':
    main()