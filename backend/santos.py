import threading

from codes.query_santos import *


class Santos:
    def __init__(self, which_benchmark=2):
        self.file_name = None
        self.return_list = []
        self.dtp, self.dfp, self.dfn = 0, 0, 0
        self.thread = None

        self.map_k = 20
        self.which_benchmark = which_benchmark
        if which_benchmark == 1:
            current_benchmark = "tus"
            self.map_k = 60
        elif which_benchmark == 2:
            current_benchmark = "santos"
            self.map_k = 10
        else:
            current_benchmark = "real_tables"

        self.which_mode = 3
        current_mode = "full"
        benchmarkLoadStart = time.time()

        # load the YAGO KB from dictionary files and data lake table names
        # edit the path below according to the location of the pickle files.
        YAGO_PATH = r"../yago/yago_pickle/"
        # edit the line below if the dlt are at different locations
        self.QUERY_TABLE_PATH = r"../benchmark/" + current_benchmark + "_benchmark/query/"

        LABEL_FILE_PATH = YAGO_PATH + "yago-wd-labels_dict.pickle"
        TYPE_FILE_PATH = YAGO_PATH + "yago-wd-full-types_dict.pickle"
        CLASS_FILE_PATH = YAGO_PATH + "yago-wd-class_dict.pickle"
        FACT_FILE_PATH = YAGO_PATH + "yago-wd-facts_dict.pickle"

        FD_FILE_PATH = r"../groundtruth/" + current_benchmark + "_FD_filedict.pickle"
        GROUND_TRUTH_PATH = r"../groundtruth/" + current_benchmark + "UnionBenchmark.pickle"
        SUBJECT_COL_PATH = r"../groundtruth/" + current_benchmark + "IntentColumnBenchmark.pickle"
        YAGO_MAIN_INVERTED_INDEX_PATH = r"../hashmap/" + current_benchmark + "_main_yago_index.pickle"
        YAGO_MAIN_RELATION_INDEX_PATH = r"../hashmap/" + current_benchmark + "_main_relation_index.pickle"
        YAGO_MAIN_PICKLE_TRIPLE_INDEX_PATH = r"../hashmap/" + current_benchmark + "_main_triple_index.pickle"
        pickle_extension = "pbz2"

        SYNTH_TYPE_LOOKUP_PATH = r"../hashmap/" + current_benchmark + "_synth_type_lookup." + pickle_extension
        SYNTH_RELATION_LOOKUP_PATH = r"../hashmap/" + current_benchmark + "_synth_relation_lookup." + pickle_extension

        SYNTH_TYPE_KB_PATH = r"../hashmap/" + current_benchmark + "_synth_type_kb." + pickle_extension
        SYNTH_RELATION_KB_PATH = r"../hashmap/" + current_benchmark + "_synth_relation_kb." + pickle_extension

        SYNTH_TYPE_INVERTED_INDEX_PATH = r"../hashmap/" + current_benchmark + "_synth_type_inverted_index." + pickle_extension
        SYNTH_RELATION_INVERTED_INDEX_PATH = r"../hashmap/" + current_benchmark + "_synth_relation_inverted_index." + pickle_extension

        # MAP_PATH = r"../stats/" + current_benchmark + "_benchmark_result_by_santos_"+current_mode+".csv"
        # FINAL_RESULT_PICKLE_PATH = r"../stats/" + current_benchmark + "_benchmark_result_by_santos_"+current_mode+".pickle"

        # TRUE_RESULTS_PATH = r"../stats/" + current_benchmark + "_benchmark_true_result_by_santos_"+current_mode+".csv"
        # FALSE_RESULTS_PATH = r"../stats/" + current_benchmark + "_benchmark_false_result_by_santos_"+current_mode+".csv"
        # QUERY_TIME_PATH = r"../stats/" + current_benchmark + "_benchmark_query_time_by_santos_"+current_mode+".csv"

        # load pickle files to the dictionary variables
        yago_loading_start_time = time.time()
        self.fd_dict = genFunc.loadDictionaryFromPickleFile(FD_FILE_PATH)
        if self.which_mode == 1 or self.which_mode == 3:
            self.label_dict = genFunc.loadDictionaryFromPickleFile(LABEL_FILE_PATH)
            self.type_dict = genFunc.loadDictionaryFromPickleFile(TYPE_FILE_PATH)
            self.class_dict = genFunc.loadDictionaryFromPickleFile(CLASS_FILE_PATH)
            self.fact_dict = genFunc.loadDictionaryFromPickleFile(FACT_FILE_PATH)
            self.yago_inverted_index = genFunc.loadDictionaryFromPickleFile(YAGO_MAIN_INVERTED_INDEX_PATH)
            yago_relation_index = genFunc.loadDictionaryFromPickleFile(YAGO_MAIN_RELATION_INDEX_PATH)
            self.main_index_triples = genFunc.loadDictionaryFromPickleFile(YAGO_MAIN_PICKLE_TRIPLE_INDEX_PATH)
        else:
            self.label_dict = {}
            self.type_dict = {}
            self.class_dict = {}
            self.fact_dict = {}
            self.yago_inverted_index = {}
            yago_relation_index = {}
            self.main_index_triples = {}

        # load synth indexes
        # synth_type_lookup = genFunc.loadDictionaryFromCsvFile(SYNTH_TYPE_LOOKUP_PATH)
        # synth_relation_lookup = genFunc.loadDictionaryFromCsvFile(SYNTH_RELATION_LOOKUP_PATH)
        if self.which_mode == 2 or self.which_mode == 3:
            self.synth_type_kb = genFunc.loadDictionaryFromPickleFile(SYNTH_TYPE_KB_PATH)
            self.synth_relation_kb = genFunc.loadDictionaryFromPickleFile(SYNTH_RELATION_KB_PATH)
            self.synth_type_inverted_index = genFunc.loadDictionaryFromPickleFile(SYNTH_TYPE_INVERTED_INDEX_PATH)
            self.synth_relation_inverted_index = genFunc.loadDictionaryFromPickleFile(SYNTH_RELATION_INVERTED_INDEX_PATH)
        else:
            self.synth_type_kb = {}
            self.synth_relation_kb = {}
            self.synth_type_inverted_index = {}
            self.synth_relation_inverted_index = {}

            # load the union groundtruth and subject columns
        self.ground_truth = genFunc.loadDictionaryFromPickleFile(GROUND_TRUTH_PATH)
        self.subject_col = genFunc.loadDictionaryFromPickleFile(SUBJECT_COL_PATH)
        benchmarkLoadEnd = time.time()
        difference = int(benchmarkLoadEnd - benchmarkLoadStart)
        print("Time taken to load benchmarks in seconds:", difference)
        print("-----------------------------------------\n")

        # This is the place where could be sepreated.

    def query_tables(self, table_name: str):
        self.return_list = []
        self.dtp, self.dfp, self.dfn = 0, 0, 0

        scoring_function = 2
        computation_start_time = time.time()
        truePositive = [0, 0, 0, 0, 0, 0, 0, 0]
        falsePositive = [0, 0, 0, 0, 0, 0, 0, 0]
        falseNegative = [0, 0, 0, 0, 0, 0, 0, 0]
        avg_pr = [[], [], [], [], [], [], [], []]
        avg_rc = [[], [], [], [], [], [], [], []]
        query_table_yielding_no_results = 0
        map_output_dict = {}
        rue_output_dict = {}
        false_output_dict = {}
        total_queries = 1
        all_query_time = {}

        precision = []
        recall = []

        for table in [table_name]:
            table_name = table.rsplit(os.sep, 1)[-1]

            print("Processing Table number:", total_queries)
            print("Table Name:", table_name)
            total_queries += 1
            if (table_name in self.ground_truth):
                expected_tables = self.ground_truth[table_name]
                totalPositive = len(expected_tables)
                k = len(expected_tables)
                value_of_k = [5, 10, 20, 30, 40, 50, 60, len(expected_tables)]
            else:
                print("The ground truth for this table is missing.")
                continue
            current_query_time_start = time.time_ns()
            bagOfSemanticsFinal = []
            input_table = pd.read_csv(table, encoding='latin1')
            unique_values = input_table.nunique().max()
            rowCardinality = {}
            rowCardinalityTotal = 0
            bag_of_semantics_final = []
            col_id = 0
            stemmed_file_name = Path(table).stem
            subject_index = self.subject_col[stemmed_file_name]
            if self.which_mode == 1 or self.which_mode == 3:
                relation_tuple_bag_of_words, entity_finding_relations, relation_dependencies, relation_dictionary, recent_hits = computeRelationSemantics(
                    input_table, subject_index, self.label_dict, self.fact_dict)
                column_tuple_bag_of_words, column_dictionary, subject_semantics = computeColumnSemantics(input_table,
                                                                                                         subject_index,
                                                                                                         self.label_dict,
                                                                                                         self.type_dict,
                                                                                                         self.class_dict,
                                                                                                         entity_finding_relations,
                                                                                                         scoring_function)
            else:
                relation_tuple_bag_of_words = []
                entities_finding_relation = {}
                relation_dependencies = []
                relation_dictionary = {}
                column_tuple_bag_of_words = []
                column_dictionary = {}
                subject_semantics = ""
            if self.which_mode == 2 or self.which_mode == 3:
                synthetic_relation_dictionary, synthetic_triples_dictionary, synth_subject_semantics = computeSynthRelation(
                    input_table, subject_index, self.synth_relation_kb)
                synthetic_column_dictionary = computeSynthColumnSemantics(input_table, self.synth_type_kb)
            else:
                synthetic_relation_dictionary = {}
                synthetic_triples_dictionary = {}
                synthetic_column_dictionary = {}
                synth_subject_semantics = set()
            current_relations = set()
            if table_name in self.fd_dict:
                current_relations = set(self.fd_dict[table_name])
            for item in relation_dependencies:
                current_relations.add(item)
            query_table_triples = {}
            synth_query_table_triples = {}
            if len(column_dictionary) > 0:
                for i in range(0, max(column_dictionary.keys())):
                    subject_type = column_dictionary.get(i, "None")
                    if subject_type != "None":
                        for j in range(i + 1, max(column_dictionary.keys()) + 1):
                            object_type = column_dictionary.get(j, "None")
                            relation_tuple_forward = "None"
                            relation_tuple_backward = "None"
                            if object_type != "None":
                                for subject_item in subject_type:
                                    for object_item in object_type:
                                        subject_name = subject_item[0]
                                        subject_score = subject_item[1]
                                        object_name = object_item[0]
                                        object_score = object_item[1]
                                        if str(i) + "-" + str(j) in current_relations:
                                            relation_tuple_forward = relation_dictionary.get(str(i) + "-" + str(j), "None")
                                        if str(j) + "-" + str(i) in current_relations:
                                            relation_tuple_backward = relation_dictionary.get(str(j) + "-" + str(i), "None")
                                        column_pairs = str(i) + "-" + str(j)
                                        if relation_tuple_forward != "None":
                                            relation_name = relation_tuple_forward[0][0]
                                            relation_score = relation_tuple_forward[0][1]
                                            triple_dict_key = subject_name + "-" + relation_name + "-" + object_name
                                            triple_score = subject_score * relation_score * object_score
                                            if triple_dict_key in query_table_triples:
                                                if triple_score > query_table_triples[triple_dict_key][0]:
                                                    query_table_triples[triple_dict_key] = (triple_score, column_pairs)
                                            else:
                                                query_table_triples[triple_dict_key] = (triple_score, column_pairs)
                                        if relation_tuple_backward != "None":
                                            relation_name = relation_tuple_backward[0][0]
                                            relation_score = relation_tuple_backward[0][1]
                                            triple_dict_key = object_name + "-" + relation_name + "-" + subject_name
                                            triple_score = subject_score * relation_score * object_score
                                            if triple_dict_key in query_table_triples:
                                                if triple_score > query_table_triples[triple_dict_key][0]:
                                                    query_table_triples[triple_dict_key] = (triple_score, column_pairs)
                                            else:
                                                query_table_triples[triple_dict_key] = (triple_score, column_pairs)
            # check if synthetic KB has found triples
            for key in synthetic_triples_dictionary:
                if len(synthetic_triples_dictionary[key]) > 0:
                    synthetic_triples = synthetic_triples_dictionary[key]
                    for synthetic_triple in synthetic_triples:
                        synthetic_triple_name = synthetic_triple[0]
                        synthetic_triple_score = synthetic_triple[1]
                        if synthetic_triple_name in synth_query_table_triples:
                            if synthetic_triple_score > synth_query_table_triples[synthetic_triple_name][0]:
                                synth_query_table_triples[synthetic_triple_name] = (synthetic_triple_score, key)
                        else:
                            synth_query_table_triples[synthetic_triple_name] = (synthetic_triple_score, key)
            query_table_triples = set(query_table_triples.items())
            synth_query_table_triples = set(synth_query_table_triples.items())

            total_triples = len(query_table_triples)

            table_count_final = {}
            eliminate_less_matching_tables = {}
            tables_containing_intent_column = {}
            # to make sure that the subject column is present
            if subject_semantics != "" and subject_semantics + "-c" in self.yago_inverted_index:
                intent_containing_tables = self.yago_inverted_index[subject_semantics + "-c"]
                for table_tuple in intent_containing_tables:
                    tables_containing_intent_column[table_tuple[0]] = 1

            already_used_column = {}
            for item in query_table_triples:
                matching_tables = self.main_index_triples.get(item[0], "None")  # checks yago inv4erted index
                if matching_tables != "None":
                    triple = item[0]
                    query_score = item[1][0]
                    col_pairs = item[1][1]
                    for data_lake_table in matching_tables:
                        dlt_name = data_lake_table[0]
                        if triple in synth_subject_semantics:
                            tables_containing_intent_column[dlt_name] = 1
                        dlt_score = data_lake_table[1]
                        total_score = query_score * dlt_score
                        if (dlt_name, col_pairs) not in already_used_column:
                            if dlt_name not in table_count_final:
                                table_count_final[dlt_name] = total_score
                            else:
                                table_count_final[dlt_name] += total_score
                            already_used_column[(dlt_name, col_pairs)] = total_score
                        else:
                            if already_used_column[(dlt_name, col_pairs)] > total_score:
                                continue
                            else:  # use better matching score
                                if dlt_name not in table_count_final:
                                    table_count_final[dlt_name] = total_score
                                else:
                                    table_count_final[dlt_name] -= already_used_column[(dlt_name, col_pairs)]
                                    table_count_final[dlt_name] += total_score
                                already_used_column[(dlt_name, col_pairs)] = total_score
            synth_col_scores = {}
            for item in synth_query_table_triples:
                matching_tables = self.synth_relation_inverted_index.get(item[0], "None")  # checks synth KB index
                if matching_tables != "None":
                    triple = item[0]
                    query_rel_score = item[1][0]
                    col_pairs = item[1][1]
                    for data_lake_table in matching_tables:
                        dlt_name = data_lake_table[0]
                        if triple in synth_subject_semantics:
                            tables_containing_intent_column[dlt_name] = 1
                        dlt_rel_score = data_lake_table[1][0]
                        dlt_col1 = data_lake_table[1][1]
                        dlt_col2 = data_lake_table[1][2]
                        query_col1 = col_pairs.split("-")[0]
                        query_col2 = col_pairs.split("-")[1]
                        dlt_col1_contents = {}
                        dlt_col2_contents = {}
                        query_col1_contents = {}
                        query_col2_contents = {}
                        if (dlt_name, dlt_col1) in self.synth_type_inverted_index:
                            dlt_col1_contents = self.synth_type_inverted_index[(dlt_name, dlt_col1)]
                        if (dlt_name, dlt_col2) in self.synth_type_inverted_index:
                            dlt_col2_contents = self.synth_type_inverted_index[(dlt_name, dlt_col2)]
                        if query_col1 in synthetic_column_dictionary:
                            query_col1_contents = synthetic_column_dictionary[query_col1]
                        if query_col2 in synthetic_column_dictionary:
                            query_col2_contents = synthetic_column_dictionary[query_col2]
                        # find intersection between dlt1 and query1

                        max_score = [0, 0, 0, 0]
                        if col_pairs + "-" + dlt_col1 + "-" + dlt_col2 in synth_col_scores:
                            total_score = synth_col_scores[
                                              col_pairs + "-" + dlt_col1 + "-" + dlt_col2] * dlt_rel_score * query_rel_score
                        else:
                            match_keys_11 = dlt_col1_contents.keys() & query_col1_contents.keys()
                            if len(match_keys_11) > 0:
                                for each_key in match_keys_11:
                                    current_score = dlt_col1_contents[each_key] * query_col1_contents[each_key]
                                    if current_score > max_score[0]:
                                        max_score[0] = current_score

                            match_keys_12 = dlt_col1_contents.keys() & query_col2_contents.keys()
                            if len(match_keys_12) > 0:
                                for each_key in match_keys_12:
                                    current_score = dlt_col1_contents[each_key] * query_col2_contents[each_key]
                                    if current_score > max_score[1]:
                                        max_score[1] = current_score

                            match_keys_21 = dlt_col2_contents.keys() & query_col1_contents.keys()

                            if len(match_keys_21) > 0:
                                for each_key in match_keys_21:
                                    current_score = dlt_col2_contents[each_key] * query_col1_contents[each_key]
                                    if current_score > max_score[2]:
                                        max_score[2] = current_score

                            match_keys_22 = dlt_col2_contents.keys() & query_col2_contents.keys()

                            if len(match_keys_22) > 0:
                                for each_key in match_keys_22:
                                    current_score = dlt_col2_contents[each_key] * query_col2_contents[each_key]
                                    if current_score > max_score[3]:
                                        max_score[3] = current_score

                            max_score = sorted(max_score, reverse=True)
                            synth_col_scores[col_pairs + "-" + dlt_col1 + "-" + dlt_col2] = max_score[0] * max_score[1]
                            total_score = query_rel_score * dlt_rel_score * max_score[0] * max_score[1]
                        if (dlt_name, col_pairs) not in already_used_column:
                            if dlt_name not in table_count_final:
                                table_count_final[dlt_name] = total_score
                            else:
                                table_count_final[dlt_name] += total_score
                            already_used_column[(dlt_name, col_pairs)] = total_score
                        else:
                            if already_used_column[(dlt_name, col_pairs)] > total_score:
                                continue
                            else:  # use better matching RSCS' score
                                if dlt_name not in table_count_final:
                                    table_count_final[dlt_name] = total_score
                                else:
                                    table_count_final[dlt_name] -= already_used_column[(dlt_name, col_pairs)]
                                    table_count_final[dlt_name] += total_score
                                already_used_column[(dlt_name, col_pairs)] = total_score

                                # to make sure that the match was because of intent column
            tables_to_throw = set()
            if len(tables_containing_intent_column) > 0:
                for shortlisted_table in table_count_final:
                    if shortlisted_table not in tables_containing_intent_column:
                        tables_to_throw.add(shortlisted_table)
            if len(tables_to_throw) > 0 and len(tables_to_throw) < len(table_count_final):
                for item in tables_to_throw:
                    del table_count_final[item]

            sortedTableList = sorted(table_count_final.items(), key=lambda x: x[1], reverse=True)
            temp_map = []
            for map_item in sortedTableList:
                temp_map.append(map_item[0])
            map_output_dict[table_name] = temp_map
            current_query_time_end = time.time_ns()
            all_query_time[table_name] = int(current_query_time_end - current_query_time_start) / 10 ** 9
            if self.which_benchmark == 3:
                continue
            print("Dynamic K = ", k)
            print("The best matching tables are:")
            i = 0
            dtp = 0
            dfp = 0
            temp_true_results = []
            temp_false_results = []
            enum = 0
            for key in sortedTableList:
                data_lake_table = key[0].split(os.sep)[-1]
                self.return_list.append(data_lake_table)
                if enum < 5:
                    print(data_lake_table, key[1])
                enum += 1
                if data_lake_table in expected_tables:
                    dtp = dtp + 1
                    temp_true_results.append(data_lake_table)
                    if enum <= 5:
                        print("true")
                else:
                    dfp = dfp + 1
                    temp_false_results.append(data_lake_table)
                    if enum <= 5:
                        print("false")
                i = i + 1
                if (i >= k):
                    break
            if dtp + dfp == 0:
                query_table_yielding_no_results += 1
            rue_output_dict[table_name] = temp_true_results
            false_output_dict[table_name] = temp_false_results
            print("Current True positives:", dtp)
            print("Current False Positives:", dfp)
            print("Current False Negative:", totalPositive - dtp)
            precision.append(dtp / (dtp + dfp))
            recall.append(dtp / (dtp + (totalPositive - dtp)))
        self.dtp, self.dfp = dtp, dfp
        self.dfn = totalPositive - dtp

        computation_end_time = time.time()
        difference = int(computation_end_time - computation_start_time)
        print("Time taken to process all query tables and print the results in seconds:", difference)
        pa = sum(precision) / len(precision)
        print("Pre Avg: ", pa)
        ra = sum(recall) / len(recall)
        print("Rec avg:", ra)

    def get_result(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        return self.return_list, self.dtp, self.dfp, self.dfn

    def start_query(self, file_name: str):
        self.file_name = file_name
        file_path = "..\\benchmark\\santos_benchmark\\query\\" + file_name
        self.thread = threading.Thread(target=self.query_tables, args=(file_path,))
        self.thread.start()
