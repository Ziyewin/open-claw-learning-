# 功能说明：编写代码，计算一个正整数的二进制间隔长度

def binary_gap_length(n: int) -> int:
    """计算正整数 n 的二进制间隔长度（最长连续0的个数）"""
    # 去掉二进制末尾的0，因为末尾的0不夹在两个1之间
    while n > 0 and n % 2 == 0:
        n >>= 1
    
    max_gap = 0
    current_gap = 0
    
    while n > 0:
        if n & 1:  # 当前位是1
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:      # 当前位是0
            current_gap += 1
        n >>= 1
    
    return max_gap

# 多组示例打印
if __name__ == "__main__":
    test_numbers = [1, 2, 5, 6, 8, 9, 15, 16, 20, 32, 65, 1041]
    for num in test_numbers:
        gap = binary_gap_length(num)
        print(f"数字 {num:5d} (二进制 {bin(num):>12s}) 的二进制间隔长度为: {gap}")