

# Music Generation with Music Theory


## Overview


This project documents the comprehensive process of fine-tuning the **MusicGen** model to generate music based on provided chord progressions. The overarching goal was to explore the extent to which a state-of-the-art deep learning model could understand and apply music theory through targeted training and fine-tuning.


During the initial phases of this project, I considered various approaches and models to achieve the desired outcome. Among them, **Audio LDM**, a latent diffusion model designed specifically for audio generation, appeared to be a promising candidate. However, after 4 weeks of in-depth analysis and preliminary experiments, I encountered significant challenges, such as the lack of a public fine-tuning loop and the limitations imposed by the small dataset available. These obstacles ultimately led me to pivot towards **MusicGen**, a more robust and flexible system that could better accommodate the project’s needs.


## MusicGen


**MusicGen** is a text and melody-conditioned music generation system developed by Meta. It stands out due to its ability to generate high-quality music based on textual descriptions or melodic inputs. You can learn more about the technical aspects of MusicGen in the official [MusicGen documentation](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md).


### Features


One of the most remarkable features of the MusicGen model is its **multi-stream encoding capability**. Multi-stream encoding is a sophisticated method used to manage complex data, such as audio or music, by decomposing it into multiple parallel streams. Each stream represents a different aspect or feature of the original signal, allowing for a more granular and detailed analysis. These streams are then encoded separately into discrete tokens, which can be efficiently processed by machine learning models.


In practical terms, the MusicGen model can process multiple simultaneous sounds, maintaining their distinct characteristics. For instance, when the model is trained on a piece of rock music, it can independently process the sound of the electronic bass and the drums as separate streams. Each stream, representing different instruments or sound characteristics, is encoded into discrete tokens that capture the unique features of those sounds. This capability allows the model to understand and generate music where the individual elements, such as the bassline and the drum pattern, can be independently controlled and manipulated, offering a high degree of flexibility in music generation.






## What I Did


### 0. Initial Exploration with Audio LDM


Before fully committing to MusicGen, I initially explored the **Audio LDM** system, which is based on latent diffusion models. Audio LDM seemed well-suited for generating music, particularly because of its advanced architecture designed to capture the latent structures in audio data. However, the system posed some significant challenges:


- **Lack of a Public Fine-Tuning Loop**:


   One of the primary issues with Audio LDM was the absence of a publicly available fine-tuning loop. Fine-tuning is a crucial step in adapting a pre-trained model to specific tasks or datasets. Without an accessible fine-tuning loop, I had to write my own, which required a deep dive into the model’s architecture and considerable effort to implement correctly.




 - **Insufficient Data**: 
    Another challenge was the relatively small size of the dataset available for training. Latent diffusion models, including Audio LDM, generally require large amounts of data to perform effectively. The limited dataset constrained the model’s ability to learn and generalize, which significantly hampered the results. Particularly because the description of each data almost only consists of chord names, the dataset did not have enough diversity that the model could understand.


These challenges led me to reconsider the approach and eventually pivot to using MusicGen, which provided a more practical and feasible framework for achieving the project’s goals.


### 1. **Data Preparation**


Now moving onto MusicGen, to effectively train the MusicGen model, I undertook a thorough data preparation process. This involved creating a dataset from an existing, royalty-free music dataset available online. The dataset creation process was crucial for ensuring that the model could learn and generate music that aligns with specific chord progressions.


The original audio file was an entire piece of music(2-5min). In order to use these audio files to train MusicGen, the dataset was segmented into smaller sections based on musical elements such as harmony, chord progressions, and structural sections (e.g., verses, choruses). This segmentation was essential to break down the music into manageable pieces that the MusicGen model could process effectively. The segments were kept within a range of 10 to 30 seconds, primarily due to the limitations imposed by the Knox environment and our GPU cluster. Longer audio sequences would have been difficult to process due to the limited GPU memory available.


- **Collection and Organization**: I began by collecting and organizing a dataset of chord progressions and their corresponding audio files. This dataset served as the foundation for the training process.
- **Manifest Creation**: To ensure the dataset was properly formatted for use with the MusicGen model, I created manifest files. These files provided a structured format that the model could easily interpret, facilitating the training process.


### 2. **Fine-Tuning MusicGen**


Once the data was prepared, I proceeded to fine-tune the MusicGen model using the segmented dataset. Fine-tuning involved adjusting the pre-trained model to better align with the specific task of generating music based on chord progressions.
- **Code Adjustments**


   In order to process this model with Knox's GPU cluster, I used the Jupyter Notebook server to conduct all of the executions. However, the models on github are not meant to be used on the Jupyter Notebook. This environmental mismatch led to a week of debugging.
- **Hyperparameter Tuning**:


   I experimented with various hyperparameters, including learning rate, batch size, and the number of epochs, to optimize the model’s performance. Each of these parameters plays a critical role in how well the model learns and generalizes from the data.
- **Dataset Adjustments**:
   Throughout the fine-tuning process, I made several adjustments to the dataset. In the first place, I was training the model with 2-6 seconds long data due to the less computational spaces. However, with these lengths, the model cuts out the after generating 3 seconds long sound. So, instead of training with those data, I increased the duration of each data into 10 to 30 seconds, which led to a stable generation process.
### 3. **Generating Music**


With the fine-tuning complete, I used the MusicGen model to generate music based on the chord names that I trained with.


### Demo


To showcase the capabilities of the fine-tuned MusicGen model, I have prepared several demo tracks.


**[Demo1 Easy Generation:]**:  
The two audio files below are generated with the same prompt: "piano CMajor". The one above is generated by the pre-trained model before fine-tuning.


- [Before fine-tuning](not_tuned_Piano_CMjor.wav)
- [After fine-tuning](fine_tuned_Piano_Cmaj.wav)




While the fine-tuned one might not sound as good as the not-fine-tuned one, it is important to note that the base note that keeps going all the time in the fine-tuned one is a note C, which is the most important note in the CMajor chord. Compared to that, a not-fine-tuned one does not have C in it.


**[Demo2 Difficult Generation:]**:  
The two audio files below are generated with the same prompt: "ukulele Cmaj, followed by Fmaj, Gmaj, and Cmaj". With this prompt, the model should generate a piece of music with the sense of progression in harmony, and the last part should sound like an end of piece. The one above is generated by the pre-trained model before fine-tuning.


This is before fine-tuning:
<audio controls src="not finetuned ukulele Cmaj, followed by Fmaj, Gmaj, and Cmaj.wav" title="Title"></audio> 

- [Before fine-tuning](not finetuned ukulele Cmaj, followed by Fmaj, Gmaj, and Cmaj.wav)
- [After fine-tuning](fine-tuned ukulele Cmaj, followed by Fmaj, Gmaj, and Cmaj.wav)


<audio controls>
 <source src="fine-tuned ukulele Cmaj, followed by Fmaj, Gmaj, and Cmaj.wav" type="audio/wav" />
 Your browser does not support the audio element.
</audio> 


In this task, the difference is a little more obvious. We have clearer ukulele sound and harmony actually changes over the time with the fine-tuned one,compared with not fine tuned one. The most important part in the fine-tuned audio is the last part.
### 4. **Challenges and Solutions**


Throughout the project, I encountered several challenges that required innovative solutions:


- **Insufficient Dataset**:
There are not a lot of audio dataset that contain both audio and its musical description such as harmony, cadences, and modulation and key. In order to overcome this problem, it might be necessary to build a classifier model that can analyze music audio so that I can create the necessary dataset for this model just from audio.
- **GPU Limitation**:
We are currently working with a 48GB GPU cluster environment. While this is quite a lot for other deep learning tasks, for music generation, this amount of GPU memory easily gets out of memory. And because of this issue, I needed to use the smallest version of MusicGen for this project. One realistic solution, other than purchasing a new GPU cluster, is to increase the amount of shared memory. Currently, in any training of machine learning models within our environment, we need to limit the "num_worker" parameter to 0, which does not allow the model to proceed parallel training.
- **Initial Challenges with Audio LDM**: As previously mentioned, my initial attempt to use the Audio LDM system highlighted the importance of having a robust and accessible model framework. The lack of a public fine-tuning loop and the small dataset size forced me to pivot to MusicGen, which proved to be a more viable solution.


### 5. **Next Steps**


The successful implementation of this project opens up several avenues for further exploration:


- **Further Fine-Tuning with Diverse Datasets**: There is potential to fine-tune the model further using different datasets, which could help in generating a broader range of musical styles and genres.
- **Evaluation Method**: Since I did not have enough time and skill to computationally evaluate the generated audio, it is necessary to come up with a way of how well the generated audio follows the given instruction.


## Reflection


This project was a significant learning experience, deepening my understanding of music generation models and the complexities involved in fine-tuning them for specific tasks. The initial challenges I faced with Audio LDM, particularly the need to create a fine-tuning loop from scratch and the limitations imposed by a small dataset, provided valuable lessons in model selection, and data preparation. Also, the experience of reading code line by line helped me deepen understanding of how latent diffusion models are working.


Ultimately, the decision to pivot to MusicGen proved to be the right choice, as it offered a more flexible and accessible framework for achieving the project’s goals. The experience highlighted the importance of having a well-prepared dataset and the ability to adapt to challenges as they arise. This project has not only enhanced my technical skills but also provided insights into the broader field of music generation and the role of machine learning in understanding and applying music theory.



