from utils import process_directory,\
    openai_wrapper,chunk_text,convert_to_networkx,\
    get_stable_connected_components,convert_to_entity_extraction_format,\
    get_embedding,client
from PromptsAndClasses.entity_extraction import EntityExtractionResponseFormat,EntityExtractionPrompt
from PromptsAndClasses.entity_correction import EntityCorrectionPrompt
from PromptsAndClasses.community_report import CommunityReportResponseFormat, CommunityReportPrompt
import json,os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
chunk_size = 500
overlap_size = 50

def single_article_init(path,data_str):
    chunk_list = chunk_text(data_str)
    print("#"*8,"chunked the text")
    
    # analyse the chunk one by one
    init_final_result = EntityExtractionResponseFormat(entities=[], relations=[])
    for chunk in chunk_list:
        current_entity_list = init_final_result.get_all_entity_names()
        input_messages = f"current entity list : {current_entity_list}, the text : {chunk}"
        
        init_chunk_result= openai_wrapper(system_messages=EntityExtractionPrompt,
                                          input_messages= input_messages,
                                          response_format=EntityExtractionResponseFormat)
        init_final_result = init_final_result.merge(init_chunk_result)
    print("#"*8, "chunked text analysed done")
    
    # give all the entity a unique describtion
    init_final_result_str = init_final_result.tostr()
    init_chunk_correction_result= openai_wrapper(system_messages=EntityCorrectionPrompt,
                                          input_messages= init_final_result_str,
                                          response_format=EntityExtractionResponseFormat)
    print("#"*8, "chunked result get corrected")
    
    init_chunk_correction_result_path = os.path.join(path,f'output/init_chunk_correction_result.json')
    with open(init_chunk_correction_result_path, 'w') as f:
        init_chunk_correction_result_str = json.dumps(init_chunk_correction_result.dict(), indent=4)
        f.write(init_chunk_correction_result_str)
        print("#"*8, f"init_chunk_correction_result_path saved at {init_chunk_correction_result_path}")
    
    # community detect based on Leiden Hierarchical Community Detection
    # graph create
    init_graph = convert_to_networkx(init_chunk_correction_result)
    
    # graph Community Detection 
    # todo length and max node for the community node detection
    community_detect_result = get_stable_connected_components(init_graph) # we only have one for this simple story
    community_detect_result = community_detect_result[0] # only one compontent make this simple
    
    # community graph to str
    init_community_sub_graph_result = convert_to_entity_extraction_format(community_detect_result)
    init_community_sub_graph_result_str = init_community_sub_graph_result.tostr()
    
    init_community_report_result= openai_wrapper(system_messages=CommunityReportPrompt,
                                          input_messages= init_community_sub_graph_result_str,
                                          response_format=CommunityReportResponseFormat)
    print("#"*8, "community report done")
    
    init_community_report_result_path = os.path.join(path,f'output/init_community_report_result.json')
    with open(init_community_report_result_path, 'w') as f:
        init_community_report_result_str = json.dumps(init_community_report_result.dict(), indent=4)
        f.write(init_community_report_result_str)
        print("#"*8, f"community report saved at {init_community_report_result_path}")
    
    # with open('init_chunk_correction_result.json', 'r') as file:
    #     init_chunk_correction_result = json.load(file)
    #     init_chunk_correction_result = EntityExtractionResponseFormat(**init_chunk_correction_result)
    # with open('init_community_report_result.json', 'r') as file:
    #     init_community_report_result = json.load(file)
    #     init_community_report_result = CommunityReportResponseFormat(**init_community_report_result)
    
    #### all the processed info stored in the init_list(all str)
    init_list = chunk_list # text chunk
    # community summary
    init_list.append(init_community_report_result.summary)
    # community findings
    for item in init_community_report_result.findings: init_list.append(item.to_str())
    # nodes
    for item in init_chunk_correction_result.entities: init_list.append(item.tostr())
    # edges
    for item in init_chunk_correction_result.relations: init_list.append(item.tostr())
    
    
    #### make embeddings here
    init_embedding_map = {}
    embeddings = get_embedding(init_list)  # Pass the entire init_list at once
    for item, embedding in zip(init_list, embeddings):
        init_embedding_map[item] = embedding
    
    init_embedding_map_path = os.path.join(path,f'output/init_embedding_map.json')
    with open(init_embedding_map_path, 'w') as f:
        json.dump(init_embedding_map, f, indent=4)
        print("#"*8, f"embedding map saved at {init_embedding_map_path}")

def database_init(path):
    path_list =  [os.path.join(path,'input'),os.path.join(path,'output')]
    for folder_check_path in path_list:
        if not os.path.exists(folder_check_path):
            os.makedirs(folder_check_path)
    
    # load the data and we only test a csv
    data = process_directory(path)  # a not empty list that contains different file lists
    data = data[0] # for test, only one file here

    single_article_init(path, data)
        
def load_init_data(path):
    
    check_file_list = [os.path.join(path, f'output/{file}') for file in os.listdir(os.path.join(path, 'output'))\
                        if file.startswith('init_embedding_map') and file.endswith('.json')]
    
    for check_file_path in check_file_list:
        if not os.path.isfile(check_file_path): 
            raise ValueError(f'{check_file_path} do not existe')
    
    # this will load all the embedding map into one embedding map
    embedding_map = {}
    for check_file_path in check_file_list:
        with open(check_file_path, 'r') as f:
            embedding_map.update(json.load(f))

    return embedding_map


def ask(input_text, embedding_map):
    # check the file 
    
    # get embedding
    input_text_embedding = get_embedding(input_text)

    # Use numpy arrays for faster cosine similarity calculation
    embeddings = np.array(list(embedding_map.values()))
    similarities = np.dot(embeddings, input_text_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_text_embedding))

    # Create a sorted list of texts based on similarity
    valid_indices = np.where(similarities > 0.5)[0]  # Get indices where similarity is greater than 0.5
    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:3]]  # Get indices of the top 3 similarities
    top3_info = " ".join(list(embedding_map.keys())[i] for i in sorted_indices)
    print(top3_info)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": f"You are a good assistant to humans. \
                    You will answer human questions based on the following information.\
                        {top3_info}"},\
                {"role": "user", "content": f"{input_text}"}],
        stream=True,
)   
    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
        yield partial_message
    
if __name__ == "__main__":
    database_init('./data-mingraphrag/')
    # load_init_data('./data-mingraphrag/')
    # database_init('./data-mingraphrag/')
    # ask('./data-mingraphrag/','the relation between cinderella and the prince')