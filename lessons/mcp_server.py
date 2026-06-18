from fastmcp import FastMCP

# 初始化MCP服务
mcp = FastMCP("Auto-Call-Tools-Server")

@mcp.tool()
def count_char(text: str) -> str:
    """统计一段文本的字符总数
    Args:
        text: 待统计的文本内容
    """
    return f"字符总数：{len(text)}"

@mcp.tool()
def concat_str(a: str, b: str) -> str:
    """拼接两段字符串
    Args:
        a: 第一个字符串
        b: 第二个字符串
    """
    return f"拼接结果：{a}{b}"

@mcp.tool()
def add_num(x: int, y: int) -> str:
    """计算两个整数的和
    Args:
        x: 第一个数字
        y: 第二个数字
    """
    return f"{x} + {y} = {x + y}"

if __name__ == "__main__":
    # 使用 stdio 模式，父子进程通信，本地自动拉起
    mcp.run()