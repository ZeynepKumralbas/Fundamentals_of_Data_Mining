import time

class eclatFile:

    def __init__(self, min_sup, minSup):

        self.min_sup = min_sup
        #    confidence = 4

        #    FreqItems = dict()

        self.data, self.trans = self.read_data('cse4063-spring2020-project-2-dataset-fpm.csv')
        self.minsup = self.trans * self.min_sup

        start = time.time()
        FreqItems = self.eclat([], sorted(self.data.items(), key=lambda item: len(item[1]), reverse=True))
        end = time.time()
        runTime = end-start

        self.runTime = runTime
        self.print_Frequent_Itemsets('eclat_output_freqitems.txt', FreqItems)

    def eclat(self, prefix, items, FreqItems=None):
        if FreqItems is None:
            FreqItems = dict()
        while items:
            i, itids = items.pop()
            isupp = len(itids)
            if isupp >= self.minsup:
                FreqItems[frozenset(prefix + [i])] = isupp/self.trans
                suffix = []
                for j, ojtids in items:
                    jtids = itids & ojtids
                    if len(jtids) >= self.minsup:
                        suffix.append((j, jtids))

                self.eclat(prefix + [i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), FreqItems)

        return FreqItems


    def print_Frequent_Itemsets(self, output_FreqItems, FreqItems):
        printed = sorted(FreqItems.items(), key=lambda item: [len(item[0]), -item[1]])
        file = open(output_FreqItems, 'w+')
        for item, support in printed:
            file.write(" {} : {} \n".format(list(item), round(support, 4)))


    def read_data(self, filename, delimiter=','):
        data = {}
        trans = 0
        with open(filename, 'r', encoding="utf8") as file:
            for row in file.read().splitlines():
                trans += 1
                for item in row.split(delimiter):
                    if item is not "":
                        if item not in data:
                            data[item] = set()
                        data[item].add(trans)
        return data, trans


    def getRunTimeList(self):
        return self.runTime