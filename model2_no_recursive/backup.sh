#!/bin/bash

filename=$1
if [ ! -f "$filename" ]; then
    echo "文件不存在: $filename"
    exit 1
fi

# 去除扩展名
base="${filename%.py}"

# 获取当前时间
now=$(date +"%Y%m%d_%H%M")

# 构造新文件名
backup_name="${base}_backup_${now}.py"

# 函数：从Python文件中移除开头的变更日志注释
remove_changelog_comments() {
    local file="$1"
    # 跳过开头的变更日志注释块
    awk '
    BEGIN { in_changelog = 0; found_start = 0 }
    /^"""/ && !found_start { 
        if (in_changelog) {
            in_changelog = 0
            found_start = 1
            next
        } else {
            in_changelog = 1
            next
        }
    }
    in_changelog { next }
    !in_changelog && found_start { print }
    !in_changelog && !found_start { 
        found_start = 1
        print 
    }
    ' "$file"
}

# 函数：生成变更日志
generate_changelog() {
    local current_file="$1"
    local previous_file="$2"
    local timestamp="$3"
    
    echo '"""'
    echo "变更日志 - 备份时间: $timestamp"
    echo "========================================"
    
    # 创建临时文件来存储去掉注释的版本
    temp_current=$(mktemp)
    temp_previous=$(mktemp)
    
    remove_changelog_comments "$current_file" > "$temp_current"
    remove_changelog_comments "$previous_file" > "$temp_previous"
    
    # 使用diff进行比较
    if diff -u "$temp_previous" "$temp_current" > /dev/null 2>&1; then
        echo "无变更"
    else
        echo "检测到以下变更:"
        echo ""
        
        # 详细的diff输出
        diff -u "$temp_previous" "$temp_current" | while IFS= read -r line; do
            case "$line" in
                "--- "*) 
                    echo "比较基准: $(basename "$previous_file")"
                    ;;
                "+++ "*) 
                    echo "当前版本: $(basename "$current_file")"
                    echo ""
                    ;;
                "@@ "*)
                    echo "位置: $line"
                    ;;
                "-"*)
                    echo "删除: ${line#-}"
                    ;;
                "+"*)
                    echo "新增: ${line#+}"
                    ;;
                " "*)
                    # 跳过上下文行，避免输出过多内容
                    ;;
            esac
        done
    fi
    
    # 清理临时文件
    rm -f "$temp_current" "$temp_previous"
    
    echo '"""'
    echo ""
}

# 查找最新的备份文件
find_latest_backup() {
    local base_name="$1"
    # 查找所有备份文件，按文件名中的时间戳排序，返回最新的
    find "$(dirname "$base_name")" -name "$(basename "$base_name")_backup_*.py" -type f 2>/dev/null | \
    sed 's/.*_backup_\([0-9]*_[0-9]*\)\.py$/\1 &/' | \
    sort -k1,1nr | \
    head -n1 | \
    cut -d' ' -f2
}

# 主要逻辑
latest_backup=$(find_latest_backup "$base")

if [ -n "$latest_backup" ] && [ -f "$latest_backup" ]; then
    echo "找到最新备份: $latest_backup"
    echo "正在生成变更日志..."
    
    # 生成带变更日志的新备份
    {
        generate_changelog "$filename" "$latest_backup" "$(date '+%Y-%m-%d %H:%M:%S')"
        remove_changelog_comments "$filename"
    } > "$backup_name"
    
    echo "已创建带变更日志的备份: $backup_name"
else
    echo "未找到之前的备份文件，创建初始备份..."
    
    # 创建初始备份，添加初始标记
    {
        echo '"""'
        echo "初始备份 - 备份时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
        echo "这是该文件的第一个备份版本"
        echo '"""'
        echo ""
        cat "$filename"
    } > "$backup_name"
    
    echo "已创建初始备份: $backup_name"
fi

echo "备份完成!"