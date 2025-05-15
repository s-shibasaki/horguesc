"""
SQL query builder for horguesc.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union

logger = logging.getLogger(__name__)

class SQLBuilder:
    """
    SQLクエリを構築するためのビルダークラス。
    複雑なSQLクエリを構造化された方法で組み立てることができます。
    """
    
    def __init__(self, base_table: str):
        """
        SQLビルダーを初期化します。
        
        Args:
            base_table: FROM句の基本テーブル名
        """
        self.base_table = base_table
        self.select_columns: List[str] = []
        self.joins: List[str] = []
        self.where_conditions: List[str] = []
        self.group_by: List[str] = []
        self.order_by: List[str] = []
        self.limit_value: Optional[int] = None
        self.parameters: List[Any] = []
        
    def select(self, *columns: str) -> 'SQLBuilder':
        """
        SELECT句に列を追加します。
        
        Args:
            *columns: 選択する列名またはSQL式
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.select_columns.extend(columns)
        return self
    
    def select_as(self, column_expr: str, alias: str) -> 'SQLBuilder':
        """
        エイリアスを付けてSELECT句に列を追加します。
        
        Args:
            column_expr: 列名またはSQL式
            alias: エイリアス名
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.select_columns.append(f"{column_expr} AS {alias}")
        return self
    
    def join(self, join_clause: str) -> 'SQLBuilder':
        """
        JOIN句を追加します。
        
        Args:
            join_clause: 完全なJOIN句
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.joins.append(join_clause)
        return self
    
    def where(self, condition: str) -> 'SQLBuilder':
        """
        WHERE句に条件を追加します。
        
        Args:
            condition: WHERE条件式
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.where_conditions.append(condition)
        return self
    
    def where_between(self, column: str, min_value: Any, max_value: Any) -> 'SQLBuilder':
        """
        BETWEEN条件を追加します。
        
        Args:
            column: 列名
            min_value: 最小値
            max_value: 最大値
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.where_conditions.append(f"{column} BETWEEN %s AND %s")
        self.parameters.extend([min_value, max_value])
        return self
    
    def where_date_range(self, date_column: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> 'SQLBuilder':
        """
        日付範囲の条件を追加します。
        
        Args:
            date_column: 日付列名
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        if start_date:
            self.where_conditions.append(f"{date_column} >= %s")
            self.parameters.append(start_date)
            
        if end_date:
            self.where_conditions.append(f"{date_column} <= %s")
            self.parameters.append(end_date)
            
        return self
    
    def where_in(self, column: str, values: List[Any]) -> 'SQLBuilder':
        """
        IN条件を追加します。
        
        Args:
            column: 列名
            values: 値のリスト
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        if not values:
            return self
            
        placeholders = ", ".join(["%s"] * len(values))
        self.where_conditions.append(f"{column} IN ({placeholders})")
        self.parameters.extend(values)
        return self
    
    def group_by_columns(self, *columns: str) -> 'SQLBuilder':
        """
        GROUP BY句に列を追加します。
        
        Args:
            *columns: グループ化する列名
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.group_by.extend(columns)
        return self
    
    def order_by_columns(self, *columns: str, desc: bool = False) -> 'SQLBuilder':
        """
        ORDER BY句に列を追加します。
        
        Args:
            *columns: 並べ替える列名
            desc: 降順かどうか
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        direction = "DESC" if desc else "ASC"
        for column in columns:
            self.order_by.append(f"{column} {direction}")
        return self
    
    def limit(self, value: int) -> 'SQLBuilder':
        """
        LIMIT句を設定します。
        
        Args:
            value: 制限する行数
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.limit_value = value
        return self
    
    def add_parameter(self, value: Any) -> 'SQLBuilder':
        """
        パラメータリストに値を追加します。
        
        Args:
            value: 追加するパラメータ値
            
        Returns:
            self: メソッドチェーン用のインスタンス
        """
        self.parameters.append(value)
        return self
        
    def build(self) -> Tuple[str, List[Any]]:
        """
        完全なSQLクエリを構築します。
        
        Returns:
            Tuple[str, List[Any]]: SQLクエリ文字列とパラメータのリスト
        """
        # SELECT句を構築
        select_clause = "SELECT " + (", ".join(self.select_columns) if self.select_columns else "*")
        
        # FROM句を構築
        from_clause = f"FROM {self.base_table}"
        
        # JOIN句を構築
        join_clause = " ".join(self.joins) if self.joins else ""
        
        # WHERE句を構築
        where_clause = ""
        if self.where_conditions:
            where_clause = "WHERE " + " AND ".join(f"({condition})" for condition in self.where_conditions)
        
        # GROUP BY句を構築
        group_by_clause = ""
        if self.group_by:
            group_by_clause = "GROUP BY " + ", ".join(self.group_by)
        
        # ORDER BY句を構築
        order_by_clause = ""
        if self.order_by:
            order_by_clause = "ORDER BY " + ", ".join(self.order_by)
        
        # LIMIT句を構築
        limit_clause = ""
        if self.limit_value is not None:
            limit_clause = f"LIMIT {self.limit_value}"
        
        # 各句を空白で連結
        query_parts = [
            select_clause,
            from_clause,
            join_clause,
            where_clause,
            group_by_clause,
            order_by_clause,
            limit_clause
        ]
        
        # 空でない部分だけを連結
        query = " ".join(part for part in query_parts if part)
        
        return query, self.parameters

    def __str__(self) -> str:
        """
        現在のビルダー状態からSQLクエリ文字列を返します。
        
        Returns:
            str: 構築されたSQLクエリ文字列
        """
        query, _ = self.build()
        return query