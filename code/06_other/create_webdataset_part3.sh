# Create image and text embeddings using the specified CLIP model.
# The input is the TAR files containing the pairs (image file name, caption) 
# from the LAION Aesthetics 6.5+ dataset.
# The output are the numpy arrays with the embeddings saved into NPY files.
#
# TODO:
# * Modify 'input_dataset' argument.
# * Modify 'output_folder'argument.
#

clip-retrieval inference \
	--input_dataset="file://OUR_DATASETS_DIR/laion_aesthetics_65p_wds/{00000..00061}.tar" \
	--output_folder=OUR_DATASETS_DIR/laion_aesthetics_65p_embeds \
	--input_format=webdataset \
	--batch_size=32 \
	--num_prepro_workers=1 \
	--enable_text=True \
	--enable_image=True \
	--enable_metadata=True \
	--wds_image_key=jpg \
	--wds_caption_key=txt \
	--clip_model="open_clip:ViT-B-32-quickgelu/laion400m_e32" \
	--use_jit=False \
	--wds_number_file_per_input_file=8192 \
	--output_partition_count=62
