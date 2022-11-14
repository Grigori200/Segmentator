from inaSpeechSegmenter import Segmenter
import pandas as pd
from typing import List
from pyannote.pipeline.typing import PipelineOutput
import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

TMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token='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')


def segmentation_to_df(segmentation: List) -> pd.DataFrame:
    df = pd.DataFrame(columns=['speaker', 'start', 'finish'])
    for segment in tqdm(segmentation, 'segmentation to dataframe'):
        speaker, start, end = segment
        df = pd.concat([df, pd.DataFrame({'speaker': speaker, 'start': start, 'finish': end}, index=[0])],
                       ignore_index=True)
    return df


def diarization_to_df(diarization: PipelineOutput) -> pd.DataFrame:
    diarized_df = pd.DataFrame(columns=['start', 'finish', 'speaker'])
    for turn, speaker, _ in tqdm(diarization.itertracks(yield_label=True), 'diarization to data frame'):
        diarized_df = pd.concat(
            [diarized_df, pd.DataFrame({'start': turn.start, 'finish': turn.end, 'speaker': speaker}, index=[0])],
            ignore_index=True)
    return diarized_df


def cut_gaps_on_borders(df: pd.DataFrame, gap_labels: List) -> pd.DataFrame:
    start_index = df[~df['speaker'].isin(gap_labels)].index[0]
    end_index = df[~df['speaker'].isin(gap_labels)].index[-1] + 1
    return df.iloc[start_index:end_index].reset_index(drop=True)


def merge_speakers(df: pd.DataFrame, speaker_labels: List, new_label: str) -> pd.DataFrame:
    tmp_df = pd.DataFrame(columns=['start', 'finish', 'speaker'])
    prev_speaker = False
    for index, row in tqdm(df.iterrows(), 'merging speakers'):
        curr_speaker = row['speaker'] in speaker_labels
        if curr_speaker and prev_speaker:
            tmp_df.loc[tmp_df.index[-1], 'finish'] = row['finish']
        else:
            tmp_df = pd.concat([tmp_df,
                                pd.DataFrame({'start': row['start'], 'finish': row['finish'], 'speaker': row['speaker']},
                                             index=[0])], ignore_index=True)
        prev_speaker = curr_speaker
    tmp_df.loc[tmp_df['speaker'].isin(speaker_labels), 'speaker'] = new_label
    return tmp_df


def remove_speakers(df: pd.DataFrame, speakers: List[str or int]) -> pd.DataFrame:
    return df.drop(df[df['speaker'].isin(speakers)].index)


def calculate_length(df: pd.DataFrame) -> pd.DataFrame:
    df['length'] = df['finish'] - df['start']
    return df


def save_segments(source_path: str, df: pd.DataFrame, destination_folder: str, save_csv: bool = False) -> pd.DataFrame:
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    part_df = pd.DataFrame(columns=['path', 'start', 'finish'])
    audio = AudioSegment.from_wav(source_path)
    core_name = os.path.basename(os.path.normpath(source_path.split('.')[0]))
    format = source_path.split('.')[-1]
    for index, segment in tqdm(df.iterrows(), 'saving segments'):
        file_path = os.path.join(destination_folder, core_name + f'_{index}.{format}')
        part_df = pd.concat([part_df, pd.DataFrame(
            {'path': file_path, 'start': segment['start'], 'finish': segment['finish']},
            index=[0])],
                            ignore_index=True, copy=False)
        audio[segment.loc['start'] * 1000:segment.loc['finish'] * 1000].export(file_path, format=format)
    if save_csv:
        part_df.to_csv(os.path.join(destination_folder, core_name + '.csv'))
    return part_df


def segmentation(file_path: str, output_folder: str, min_gap: float) -> pd.DataFrame:
    seg = Segmenter()
    segments = seg(file_path)
    df = segmentation_to_df(segments)
    gaps = ['noEnergy', 'noise', 'music']
    gap = 'gap'
    df = merge_speakers(df, gaps, gap)
    df = add_margin_of_mistake(df, min_gap, [gap])
    df = remove_speakers(df, [gap])
    tmp_df = save_segments(file_path, df, output_folder)
    return tmp_df


def add_margin_of_mistake(df: pd.DataFrame, min_gap: float, gaps: List[str]) -> pd.DataFrame:
    for (index_1, row_1), (index_2, row_2) in zip(df[:-1].iterrows(), df[1:].iterrows()):
        if row_1['speaker'] not in gaps and row_2['speaker'] in gaps:
            if row_2['finish'] - row_2['start'] < min_gap:
                df.at[index_1, 'finish'] = row_2['finish']
            else:
                df.at[index_1, 'finish'] = row_1['finish'] + min_gap
    return df


def connect_records(df: pd.DataFrame, length_limit: float = 30.) -> pd.DataFrame:
    start = df.at[0, 'start']
    end = df.at[0, 'finish']
    new_df = pd.DataFrame(columns=['start', 'finish'])
    for index, row in df.iloc[1:].iterrows():
        if row['finish'] - start < length_limit and (end == row['start'] or end == -1.):
            end = row['finish']
        else:
            new_df = pd.concat([new_df, pd.DataFrame({'start': start, 'finish': end}, index=[0])], ignore_index=True)
            start = row['start']
            end = row['finish']
    new_df = pd.concat([new_df, pd.DataFrame({'start': start, 'finish': end}, index=[0])], ignore_index=True)
    return new_df


def split_simultaneous_speakers(df: pd.DataFrame) -> pd.DataFrame:
    last_end = 0.
    for index, row in tqdm(df.iterrows(), 'splitting simultaneous speakers'):
        if row['finish'] <= last_end != 0 and row['start'] <= last_end:
            df.loc[index, 'speaker'] = -1
        df.loc[index, 'start'] = last_end
        last_end = max(last_end, row['finish'])
    return remove_speakers(df, [-1])


def diarization(file_path: str, output_folder: str, length: float) -> pd.DataFrame:
    try:
        segments = PIPELINE(file_path)
        df = diarization_to_df(segments)
    except ValueError:
        df = pd.DataFrame(columns=['start', 'finish', 'speaker'])
    df = split_simultaneous_speakers(df)
    if len(df.index) < 1:
        df = pd.concat(
            [df, pd.DataFrame({'start': 0., 'finish': length, 'speaker': 'Unknown'}, index=[0])],
            ignore_index=True)
    tmp_df = save_segments(file_path, df, output_folder)
    return tmp_df


def prepare_splits(file_path: str, start: float, end: float, min_silence_len: int = 700, silence_thresh: int = -12,
                   length_limit: float = 30.) -> pd.DataFrame:
    format = file_path.split('.')[-1]
    audio = AudioSegment.from_wav(file_path)[start * 1000:end * 1000]
    audio.export(file_path, format=format)
    return count_chunk_times(
        connect_chunks(split_recording(audio, min_silence_len, silence_thresh, length_limit), length_limit), start, end)


def split_recording(audio: AudioSegment, min_silence_len: int = 500, silence_thresh: int = -12,
                    length_limit: float = 30.) -> List[AudioSegment]:
    audio_chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True
    )
    if not audio_chunks:
        audio_chunks = split_recording(audio, min_silence_len, silence_thresh * 3 // 2, length_limit)
    tmp_chunks = []
    for chunk in tqdm(audio_chunks, 'splitting recording'):
        if chunk.duration_seconds > length_limit:
            tmp_chunks += split_recording(chunk, min_silence_len * 3 // 4, silence_thresh * 3 // 2, length_limit)
        else:
            tmp_chunks += [chunk]
    return tmp_chunks


def connect_chunks(chunks: List[AudioSegment], length_limit: float = 30.) -> List[AudioSegment]:
    segments = []
    combined = AudioSegment.empty()
    for chunk in tqdm(chunks, 'combining chunks'):
        if combined.duration_seconds + chunk.duration_seconds > length_limit:
            segments += [combined]
            combined = AudioSegment.empty()
        combined += chunk
    segments += [combined]
    return segments


def count_chunk_times(chunks: List[AudioSegment], start: float, end: float) -> pd.DataFrame:
    df = pd.DataFrame(columns=['start', 'finish'])
    last_end = start
    for chunk in tqdm(chunks, 'counting chunk times'):
        new_end = last_end + chunk.duration_seconds
        df = pd.concat([df, pd.DataFrame({'start': last_end, 'finish': new_end}, index=[0])], ignore_index=True)
        last_end = new_end
    df.loc[df.index[-1], 'finish'] = end
    return df


def limiting_length(df: pd.DataFrame, length_limit: float, base_start: float, base_end: float) -> pd.DataFrame:
    df = calculate_length(df)
    new_df = pd.DataFrame(columns=['start', 'finish'])
    for index, row in tqdm(df.iterrows(), 'length verification'):
        if row['length'] > length_limit:
            tmp_df = prepare_splits(row['path'], row['start'], row['finish'], length_limit=length_limit)
            new_df = pd.concat([new_df, tmp_df], ignore_index=True)
        else:
            new_df = pd.concat([new_df,
                                pd.DataFrame({'start': row['start'], 'finish': row['finish']},
                                             index=[0])], ignore_index=True)
    new_df['start'] += base_start
    new_df['finish'] += base_start
    new_df.at[len(new_df) - 1, 'finish'] = base_end
    return new_df


def delete_tmp_data(df: pd.DataFrame) -> None:
    for index, row in tqdm(df.iterrows(), 'deleting tmp data'):
        if os.path.exists(row['path']):
            os.remove(row['path'])


def save_results(source_path: str, df: pd.DataFrame, destination_folder: str) -> str:
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    part_df = pd.DataFrame(columns=['recording', 'start', 'finish'])
    audio = AudioSegment.from_wav(source_path)
    core_name = os.path.basename(os.path.normpath(source_path.split('.')[0]))
    subfolder = os.path.join(destination_folder, core_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    df.to_csv(os.path.join(subfolder, 'timestamps.csv'))
    format = source_path.split('.')[-1]
    for index, segment in tqdm(df.iterrows(), 'saving segments'):
        file_path = os.path.join(subfolder, f'{index}.wav')
        part_df = pd.concat(
            [part_df,
             pd.DataFrame({'recording': index, 'start': segment['start'], 'finish': segment['finish']}, index=[0])],
            ignore_index=True, copy=False)
        audio[segment.loc['start'] * 1000:segment.loc['finish'] * 1000].export(file_path, format=format)
    part_df.to_csv(os.path.join(subfolder, 'timestamps.csv'))
    return subfolder


def full_segmentation(file_path: str, output_folder: str, length_limit: float, min_gap: float) -> str:
    full_df = pd.DataFrame(columns=['start', 'finish'])
    df = segmentation(file_path, TMP_FOLDER, min_gap)
    for index, row in tqdm(df.iterrows(), 'iteration over segments'):
        diar_df = diarization(row['path'], TMP_FOLDER, row['finish'] - row['start'])
        if len(diar_df.index) > 0:
            limited_df = limiting_length(diar_df, length_limit, row['start'], row['finish'])
            full_df = pd.concat([full_df, limited_df], ignore_index=True)
            delete_tmp_data(diar_df)
    delete_tmp_data(df)
    full_df = connect_records(full_df)
    return save_results(file_path, full_df, output_folder)


def segmentation_without_diarization(file_path: str, output_folder: str, length_limit: float, min_gap: float) -> str:
    df = segmentation(file_path, TMP_FOLDER, min_gap)
    limited_df = limiting_length(df, length_limit, df.at[0, 'start'], df.at[len(df.index)-1, 'finish'])
    delete_tmp_data(df)
    full_df = connect_records(limited_df)
    return save_results(file_path, full_df, output_folder)


def sole_limitation(file_path: str, output_folder: str, length_limit: float, min_gap: float) -> str:
    audio = AudioSegment.from_wav(file_path)
    df = pd.DataFrame({'path': [file_path], 'start': [0.], 'finish': [audio.duration_seconds]})
    limited_df = limiting_length(df, length_limit, 0., audio.duration_seconds)
    full_df = connect_records(limited_df)
    return save_results(file_path, full_df, output_folder)
