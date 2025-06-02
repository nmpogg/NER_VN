import re
from datetime import datetime, timedelta


class DateExtractor:
    def __init__(self):
        # Từ điển các từ khóa thời gian tương đối cố định
        self.relative_time_dict = {
            "hôm nay": 0,
            "ngày hôm nay": 0,
            "bây giờ": 0,
            "hiện tại": 0,
            "hôm qua": -1,
            "ngày hôm qua": -1,
            "ngày mai": 1,
            "ngày kia": 2,
            "hôm kia": 2,
            "tuần trước": -7,
            "tuần sau": 7,
            "tháng trước": -30,
            "tháng sau": 30,
            "năm trước": -365,
            "năm sau": 365,
            "thế kỉ trước": -36500,  # Ước lượng 100 năm
            "thế kỷ trước": -36500,  # Hỗ trợ cả 'kỉ' và 'kỷ'
            "thế kỉ sau": 36500,
            "thế kỷ sau": 36500
        }

        # Từ điển các tháng
        self.month_dict = {
            "tháng một": 1, "tháng 1": 1, "tháng giêng": 1,
            "tháng hai": 2, "tháng 2": 2,
            "tháng ba": 3, "tháng 3": 3,
            "tháng bốn": 4, "tháng 4": 4, "tháng tư": 4,
            "tháng năm": 5, "tháng 5": 5,
            "tháng sáu": 6, "tháng 6": 6,
            "tháng bảy": 7, "tháng 7": 7,
            "tháng tám": 8, "tháng 8": 8,
            "tháng chín": 9, "tháng 9": 9,
            "tháng mười": 10, "tháng 10": 10,
            "tháng mười một": 11, "tháng 11": 11,
            "tháng mười hai": 12, "tháng 12": 12,
        }

        # Đơn vị thời gian và số ngày tương ứng
        self.time_unit_dict = {
            "ngày": 1,
            "tuần": 7,
            "tháng": 30,  # Ước lượng
            "quý": 90,  # Ước lượng
            "năm": 365,  # Ước lượng
            "thế kỉ": 36500,  # Ước lượng 100 năm
            "thế kỷ": 36500,  # Hỗ trợ cả 'kỉ' và 'kỷ'
        }

    def extract_date_entity(self, tokens, ner_tags):
        """Trích xuất các thực thể DATE từ chuỗi token và tag NER"""
        date_entities = []
        current_date = []
        current_indices = []

        for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
            if tag == 'B-DATE':
                if current_date:  # Kết thúc thực thể trước đó nếu có
                    date_entities.append((' '.join(current_date), current_indices))
                    current_date = []
                    current_indices = []
                current_date.append(token)
                current_indices.append(i)
            elif tag == 'I-DATE' and current_date:  # Tiếp tục thực thể hiện tại
                current_date.append(token)
                current_indices.append(i)
            elif current_date:  # Kết thúc thực thể hiện tại
                date_entities.append((' '.join(current_date), current_indices))
                current_date = []
                current_indices = []

        # Kiểm tra nếu còn thực thể cuối cùng chưa được thêm vào
        if current_date:
            date_entities.append((' '.join(current_date), current_indices))

        return date_entities

    def convert_to_date_format(self, date_text):
        """Chuyển đổi thực thể ngày thành định dạng ngày/tháng/năm"""
        date_text = date_text.lower().strip()
        today = datetime.now()

        # Kiểm tra xem có phải là thời gian tương đối cố định không
        for key, days_delta in self.relative_time_dict.items():
            if date_text == key or date_text.endswith(key):
                result_date = today + timedelta(days=days_delta)
                return result_date.strftime("%d/%m/%Y")

        # Xử lý biểu thức thời gian tương đối phức tạp: "X [đơn vị thời gian] trước/sau"
        pattern_relative = r"(\d+)\s+(ngày|tuần|tháng|quý|năm|thế kỉ|thế kỷ)\s+(trước|sau)"
        match = re.search(pattern_relative, date_text)
        if match:
            amount, unit, direction = match.groups()
            amount = int(amount)
            days_per_unit = self.time_unit_dict.get(unit, 1)
            days_delta = amount * days_per_unit

            if direction == "trước":
                days_delta = -days_delta

            result_date = today + timedelta(days=days_delta)
            return result_date.strftime("%d/%m/%Y")

        # Kiểm tra mẫu "ngày X tháng Y năm Z"
        pattern1 = r"ngày\s+(\d+)\s+tháng\s+(\d+)(?:\s+năm\s+(\d+))?"
        match = re.search(pattern1, date_text)
        if match:
            day, month, year = match.groups()
            day = int(day)
            month = int(month)
            year = int(year) if year else today.year
            # Xử lý năm 2 chữ số
            if year < 100:
                year += 2000
            try:
                return datetime(year, month, day).strftime("%d/%m/%Y")
            except ValueError:
                pass  # Ngày không hợp lệ

        # Kiểm tra mẫu "X/Y/Z"
        pattern2 = r"(\d+)[/-](\d+)(?:[/-](\d+))?"
        match = re.search(pattern2, date_text)
        if match:
            day, month, year = match.groups()
            day = int(day)
            month = int(month)
            year = int(year) if year else today.year
            if year < 100:
                year += 2000
            try:
                return datetime(year, month, day).strftime("%d/%m/%Y")
            except ValueError:
                # Thử đảo ngược ngày và tháng nếu không hợp lệ
                try:
                    return datetime(year, day, month).strftime("%d/%m/%Y")
                except ValueError:
                    pass

        # Kiểm tra tên tháng
        for month_name, month_num in self.month_dict.items():
            if month_name in date_text:
                # Tìm ngày trong chuỗi
                day_match = re.search(r"(\d+)\s*(?:" + month_name + ")", date_text)
                if day_match:
                    day = int(day_match.group(1))
                else:
                    # Nếu không tìm thấy ngày, giả định là ngày 1
                    day = 1

                # Tìm năm trong chuỗi
                year_match = re.search(r"năm\s+(\d+)", date_text)
                if year_match:
                    year = int(year_match.group(1))
                    if year < 100:
                        year += 2000
                else:
                    year = today.year

                try:
                    return datetime(year, month_num, day).strftime("%d/%m/%Y")
                except ValueError:
                    pass

        # Nếu không khớp với bất kỳ mẫu nào, trả về chuỗi gốc
        return date_text

    def process_ner_results(self, tagged_words):
        """Xử lý kết quả NER và thay thế thực thể DATE bằng định dạng ngày/tháng/năm
        Args:
            tagged_words: List of tuples containing (word, tag) pairs
        Returns:
            List of tuples containing (word, tag) pairs with dates formatted
        """
        date_entities = []
        current_date = []
        current_indices = []
        current_tags = []

        # Extract date entities from tagged words
        for i, (word, tag) in enumerate(tagged_words):
            if tag == 'B-DATE':
                if current_date:  # End previous entity if exists
                    date_entities.append((' '.join(current_date), current_indices, current_tags))
                    current_date = []
                    current_indices = []
                    current_tags = []
                current_date.append(word)
                current_indices.append(i)
                current_tags.append(tag)
            elif tag == 'I-DATE' and current_date:  # Continue current entity
                current_date.append(word)
                current_indices.append(i)
                current_tags.append(tag)
            elif current_date:  # End current entity
                date_entities.append((' '.join(current_date), current_indices, current_tags))
                current_date = []
                current_indices = []
                current_tags = []

        # Check if there's a final entity to add
        if current_date:
            date_entities.append((' '.join(current_date), current_indices, current_tags))

        # Create a copy of tagged words to modify
        result_words = tagged_words.copy()

        # Replace DATE entities with formatted dates
        for date_text, indices, tags in date_entities:
            formatted_date = self.convert_to_date_format(date_text)

            # Debug
            print(f"Date entity: '{date_text}' -> '{formatted_date}'")

            # Replace first token with formatted date but keep original tag
            if indices:
                result_words[indices[0]] = (formatted_date, tags[0])

                # Set remaining tokens to empty strings but keep original tags
                for idx, tag in zip(indices[1:], tags[1:]):
                    result_words[idx] = ('', tag)

        # Filter out empty tokens and return the tagged words
        return [(word, tag) for word, tag in result_words if word]


# Ví dụ sử dụng
def example():
    extractor = DateExtractor()

    # Ví dụ 1: "Hôm nay"
    text1 = [('Tôi', 'O'), ('có', 'O'), ('cuộc', 'O'), ('họp', 'O'), ('vào', 'O'), ('hôm', 'B-DATE'), ('nay', 'I-DATE')]
    result1 = extractor.process_ner_results(text1)
    print(f"Input: {text1}")
    print(f"Output: {result1}")
    print()

    # Ví dụ 2: "Hôm qua"
    text2 = [('Hôm', 'B-DATE'), ('qua', 'I-DATE'), ('tôi', 'O'), ('đã', 'O'), ('đi', 'O'), ('học', 'O')]
    result2 = extractor.process_ner_results(text2)
    print(f"Input: {text2}")
    print(f"Output: {result2}")
    print()

    # Ví dụ 3: "10 năm trước"
    text3 = [('Tôi', 'O'), ('đã', 'O'), ('tốt', 'O'), ('nghiệp', 'O'), ('đại', 'O'), ('học', 'O'), ('10', 'B-DATE'), ('năm', 'I-DATE'), ('trước', 'I-DATE')]
    result3 = extractor.process_ner_results(text3)
    print(f"Input: {text3}")
    print(f"Output: {result3}")
    print()

    # Ví dụ 4: "2 ngày trước"
    text4 = [('2', 'B-DATE'), ('ngày', 'I-DATE'), ('trước', 'I-DATE'), ('tôi', 'O'), ('đã', 'O'), ('đặt', 'O'), ('vé', 'O'), ('máy', 'O'), ('bay', 'O')]
    result4 = extractor.process_ner_results(text4)
    print(f"Input: {text4}")
    print(f"Output: {result4}")
    print()

    # Ví dụ 5: "năm trước"
    text5 = [('Thứ', 'B-DATE'), ('bảy', 'I-DATE'), ('tuần', 'I-DATE'), ('trước', 'I-DATE'), ('tôi', 'O'), ('đã', 'O'), ('đi', 'O'), ('du', 'O'), ('lịch', 'O'), ('châu', 'O'), ('Âu', 'O')]
    result5 = extractor.process_ner_results(text5)
    print(f"Input: {text5}")
    print(f"Output: {result5}")
    print()

    # Ví dụ 6: "năm sau"
    text6 = [('Tôi', 'O'), ('sẽ', 'O'), ('tốt', 'O'), ('nghiệp', 'O'), ('vào', 'O'), ('năm', 'B-DATE'), ('sau', 'I-DATE')]
    result6 = extractor.process_ner_results(text6)
    print(f"Input: {text6}")
    print(f"Output: {result6}")
    print()

    # Ví dụ 7: "thế kỉ trước"
    text7 = [('Tòa', 'O'), ('nhà', 'O'), ('này', 'O'), ('được', 'O'), ('xây', 'O'), ('dựng', 'O'), ('từ', 'O'), ('thế', 'B-DATE'), ('kỉ', 'I-DATE'), ('trước', 'I-DATE')]
    result7 = extractor.process_ner_results(text7)
    print(f"Input: {text7}")
    print(f"Output: {result7}")
    print()

    # Ví dụ 8: "thế kỉ sau"
    text8 = [('Công', 'O'), ('nghệ', 'O'), ('này', 'O'), ('sẽ', 'O'), ('thay', 'O'), ('đổi', 'O'), ('thế', 'O'), ('giới', 'O'), ('vào', 'O'), ('thế', 'B-DATE'), ('kỉ', 'I-DATE'), ('sau', 'I-DATE')]
    result8 = extractor.process_ner_results(text8)
    print(f"Input: {text8}")
    print(f"Output: {result8}")
    print()


if __name__ == "__main__":
    example()