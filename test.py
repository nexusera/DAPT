
#集合n个元素，输出所有子集
nums = [1,2,3]


def get_subsets(nums):
    res =[]
    path =[]
    n = len(nums)
    def dfs(start,k):
        #k为路径长度
        if len(path)==k:
            res.append(path[:])
            return

        for i in range(start,n):
            path.append(nums[i])
            dfs(i+1,k)
            path.pop()

    for k in range(n,-1,-1):
        dfs(0,k)
    return res
subsets = get_subsets(nums)

for s in subsets:
    print(s)