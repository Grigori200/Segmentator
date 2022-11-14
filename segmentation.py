import pandas as pd
from typing import List
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm


def calculate_length(df: pd.DataFrame) -> pd.DataFrame:
    df['length'] = df['finish'] - df['start']
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


def sole_limitation(file_path: str, output_folder: str, length_limit: float) -> str:
    audio = AudioSegment.from_wav(file_path)
    df = pd.DataFrame({'path': [file_path], 'start': [0.], 'finish': [audio.duration_seconds]})
    limited_df = limiting_length(df, length_limit, 0., audio.duration_seconds)
    full_df = connect_records(limited_df)
    return save_results(file_path, full_df, output_folder)
