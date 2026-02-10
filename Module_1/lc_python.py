class Solutions:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}

        for i, num in enumerate(nums):
            if target - num in hash_map:
                return [i,hash_map[target-num]]
            hash_map[num] = i 

    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums))!= len(nums)

    def maxProfit(self, prices: List[int]) -> int:
        minVal = prices[0]
        ans = 0
        for x in prices:
            ans = max(ans, x-minVal)
            minVal = min(minVal,x)

        return ans

    def isAnagram(self, s: str, t: str) -> bool:
        if len(t) != len(s):
            return False
        freq = {}
        for c in s:
            freq[c] = freq.get(c,0) + 1
        for c in t:
            if c not in freq or freq[c] == 0:
                return False
            freq[c]-=1

        return True

    def isValid(self, s: str) -> bool:
        a = []
        for c in s:
            if c in ['(','{','[']:
                a.append(c)
            else:
                if not a:
                    return False
                top = a.pop()
                if c == '}' and top!='{':
                    return False
                if c == ')' and top!='(':
                    return False
                if c == ']' and top!='[':
                    return False
        
        return len(a) == 0

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        ans = []
        prod = 1
        count = 0
        for x in nums:
            if x != 0:
                prod = prod * x
            else:
                count = count + 1

        if count > 1:
            return [0] * len(nums)
        
        for x in nums:
            if count == 1:
                if x == 0:
                    ans.append(prod)
                else:
                    ans.append(0)
            else:
                ans.append(prod//x)

        return ans
    
    def maxSubArray(self, nums: List[int]) -> int:
        csum = 0
        ans = float('-inf')
        for x in nums:
            if csum < 0:
                csum = 0
            csum += x
            ans = max(csum, ans)

        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        for i in range(len(nums)-2):
            if i > 0 and nums[i]==nums[i-1]:
                continue
            l,r = i+1,len(nums)-1
            while l < r:
                s = nums[i]+nums[l]+nums[r]
                if s == 0:
                    ans.append([nums[i],nums[l],nums[r]])
                    l+=1
                    r-=1
                    while l<r and nums[l]==nums[l-1]:
                        l+=1
                    while l<r and nums[r]==nums[r+1]:
                        r-=1
                elif s>0:
                    r-=1
                else:
                    l+=1
                    
        return ans

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        res = []

        for s, e in intervals:
            if not res or s > res[-1][1]:
                res.append([s, e])
            else:
                res[-1][1] = max(res[-1][1], e)

        return res

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        seen = {}
        
        for word in strs:
            key = "".join(sorted(word))
            if key in seen:
                seen[key].append(word)
            else:
                seen[key]=[word]

        ans = []
        for v in seen.values():
            ans.append(v)

        return ans

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        ptr = head
        node = None
        while(ptr != None):
            curr = ptr
            ptr = ptr.next
            curr.next = node
            node = curr

        return node

    def maxArea(self, height: List[int]) -> int:
        ans = float("-inf")
        l,r = 0,len(height)-1

        while(l<=r):
            ans = max(ans,min(height[l],height[r])*(r-l))

            if(height[r]<height[l]):
                r-=1
            else:
                l+=1
        
        return ans

    def findMin(self, nums: List[int]) -> int:
        l,r = 0,len(nums)-1

        while(l<=r):
            mid = (l+r)//2
            if nums[mid] >= max(nums[l],nums[r]):
                l = mid+1
            else:
                r = mid
                
        return nums[r]

    def characterReplacement(self, s: str, k: int) -> int:
        l,r,frq,ans = 0,0,0,0
        count = [0] * 26
        while r < len(s):
            count[ord(s[r]) - ord('A')]+=1
            frq = max(frq,count[ord(s[r]) - ord('A')])

            if (r-l+1-frq) <= k:
                ans = max(ans,r-l+1)
                r+=1
            else:
                count[ord(s[l]) - ord('A')]-=1
                l+=1
                r+=1
        
        return ans

    def lengthOfLongestSubstring(self, s: str) -> int:
        l,r,ans = 0,0,0
        seen = set()
        while r<len(s):
            while s[r] in seen:
                seen.remove(s[l])
                l+=1
            
            seen.add(s[r])
            ans = max(ans, r-l+1)
            
            r+=1
        return ans

    def numIslands(self, grid: List[List[str]]) -> int:
        self.m = len(grid)
        self.n = len(grid[0])
        count = 0

        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] == "1":
                    count += 1
                    self.dfs(grid, i, j)

        return count

    def dfs(self, grid, i, j):
        if i < 0 or j < 0 or i >= self.m or j >= self.n or grid[i][j] != "1":
            return

        grid[i][j] = "#"

        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)
    
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0,head)
        fptr = sptr = dummy
        for _ in range(n):
            fptr = fptr.next
        
        while fptr.next:
            fptr = fptr.next
            sptr = sptr.next

        sptr.next = sptr.next.next 
        return dummy.next

    
