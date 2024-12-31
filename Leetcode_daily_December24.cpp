#include<bits/stdc++.h>
#include "print_arr.h"
using namespace std;
class P_1346 {
    /*Given an array arr of integers, check if there exist two indices i and j such that :
        i != j
        0 <= i, j < arr.length
        arr[i] == 2 * arr[j]
    Example 1:  Input: arr = [10,2,5,3]
                Output: true
                Explanation: For i = 0 and j = 2, arr[i] == 10 == 2 * 5 == 2 * arr[j]

    Example 2:  Input: arr = [3,1,7,11]
                Output: false
                Explanation: There is no i and j that satisfy the conditions.*/
public:
   bool checkIfExist(vector<int> &arr){
        unordered_set<int> s;
        for (int i : arr){
            if (s.count(2 * i) || (i % 2 == 0 && s.count(i/2))) return true;
            
            s.insert(i);
        }
        return false;
    }
  /*  bool checkIfexist(vector<int>&ar){
        for(int i = 0;i < ar.size();i++){
            for(int j = 1;j < ar.size();j++){
                if(i != j  && ar[i] == 2 * ar[j]) return true;
            }
        }
        return false;
    }*/
};
class P_1455 {
    /*Given a sentence that consists of some words separated by a single space, and a searchWord, check if searchWord is a prefix of any word in sentence.
    Return the index of the word in sentence (1-indexed) where searchWord is a prefix of this word. If searchWord is a prefix of more than one word, return the index 
    of the first word (minimum index). If there is no such word return -1.
    Example 1:  Input: sentence = "i love eating burger", searchWord = "burg"
                Output: 4
                Explanation: "burg" is prefix of "burger" which is the 4th word in the sentence.
    Example 2:  Input: sentence = "this problem is an easy problem", searchWord = "pro"
                Output: 2
                Explanation: "pro" is prefix of "problem" which is the 2nd and the 6th word in the sentence, but we return 2 as it's the minimal index.*/
    public:
    int mysol(string sentence,string searchWord){ // Add trailing spaces
        int idx = -1, wc = 0;
        string my_sent = " " + sentence, my_word = " " + searchWord;
        idx = my_sent.find(my_word);
        if (idx == -1)
            return -1;
        for (int i = idx; i >= 0; i--)
        {
            if (my_sent[i] == ' ')
                wc++;
        }

        return wc;
    }
        int isPrefixOfWord(string sentence, string searchWord){//2 - ptr    
         int cur_pos = 1,n = sentence.size(),cur_idx = 0;
         while(cur_idx < n){
            while(cur_idx < n && sentence[cur_idx] == ' '){//skip leading 0's
                cur_idx++;
                cur_pos++;
            }
         
         int match = 0;
         while(cur_idx < n && match < searchWord.size() && sentence[cur_idx] == searchWord[match]){
            match++;
            cur_idx++;
         }
         if(match == searchWord.size()) return cur_pos;
         while(cur_idx < n && sentence[cur_idx] != ' '){
            cur_idx++;
         }
        }
        return -1;
    }
};
class P_2109 {
    /*You are given a 0-indexed string s and a 0-indexed integer array spaces that describes the indices in the original string where spaces will be added. 
    Each space should be inserted before the character at the given index. For example, given s = "EnjoyYourCoffee" and spaces = [5, 9], 
    we place spaces before 'Y' and 'C', which are at indices 5 and 9 respectively. Thus, we obtain "Enjoy Your Coffee".
    Return the modified string after the spaces have been added.
    Example 1:  Input: s = "LeetcodeHelpsMeLearn", spaces = [8,13,15]
                Output: "Leetcode Helps Me Learn"
    Explanation:The indices 8, 13, and 15 correspond to the underlined characters in "LeetcodeHelpsMeLearn".We then place spaces before those characters.
    Example 2:  Input: s = "icodeinpython", spaces = [1,5,7,9]
                Output: "i code in py thon"
    Explanation:The indices 1, 5, 7, and 9 correspond to the underlined characters in "icodeinpython".We then place spaces before those characters.*/
public:
    string addSpaces(string s, vector<int> &spaces){
        string ans = "";
        int i = 0,j = 0;
        while(i < s.size()){
            if(i == spaces[j] && j < spaces.size()){
                ans += ' ';
                j++;
            }
            ans += s[i];
            i++;
        }
        return ans;
    }
};
class P_2825 {
    /*You are given two 0-indexed strings str1 and str2. In an operation, you select a set of indices in str1, and for each index i in the set, increment str1[i] to the next character cyclically.
     That is 'a' becomes 'b', 'b' becomes 'c', and so on, and 'z' becomes 'a'. Return true if it is possible to make str2 a subsequence of str1 by performing the operation at most once, and false otherwise.
    Note: A subsequence of a string is a new string that is formed from the original string by deleting some (possibly none) of the characters without disturbing the relative positions of the remaining characters.
Example 1:  Input: str1 = "abc", str2 = "ad"
            Output: true
            Explanation: Select index 2 in str1.Increment str1[2] to become 'd'.Hence, str1 becomes "abd" and str2 is now a subsequence. Therefore, true is returned.

Example 2:  Input: str1 = "ab", str2 = "d"
            Output: false
            Explanation: In this example, it can be shown that it is impossible to make str2 a subsequence of str1 using the operation at most once. Therefore, false is returned.*/
public:
    bool canMakeSubsequence(string str1, string str2) {
        int i = 0,j = 0;
        while(i < str1.size()){
            if(str1[i] == str2[j] || str1[i] + 1 == str2[j]){
                j++;
            }
            else if(str1[i] == 'z' && str2[j] == 'a'){j++;}
            i++;
            if(j == str2.size()) return true;
        }
        return false;
    }
};
class P_2337 {
    /*You are given two strings start and target, both of length n. Each string consists only of the characters 'L', 'R', and '_' where: The characters 'L' and 'R' represent pieces, where a piece 'L' can move to the left only if there is a blank space directly to its left, and a piece 'R' can move to the right only if there is a blank space directly to its right.
    The character '_' represents a blank space that can be occupied by any of the 'L' or 'R' pieces.Return true if it is possible to obtain the string target by moving the pieces of the string start any number of times. Otherwise, return false.
    Example 1:  Input: start = "_L__R__R_", target = "L______RR"
                Output: true
                Explanation: We can obtain the string target from start by doing the following moves:
                - Move the first piece one step to the left, start becomes equal to "L___R__R_".
                - Move the last piece one step to the right, start becomes equal to "L___R___R".
                - Move the second piece three steps to the right, start becomes equal to "L______RR".
                Since it is possible to get the string target from start, we return true.

    Example 2:  Input: start = "R_L_", target = "__LR"
                Output: false
                Explanation: The 'R' piece in the string start can move one step to the right to obtain "_RL_".
                After that, no pieces can move anymore, so it is impossible to obtain the string target from start.

    Example 3:  Input: start = "_R", target = "R_"
                Output: false
                Explanation: The piece in the string start can move only to the right, so it is impossible to obtain the string target from start.*/
public:
    bool canChange(string start, string target){
        int n = start.size(),l = 0,r = 0;
        for(int i = 0;i < n;i++){
            if(start[i] == 'L') {
                l++;
                if(r != 0) return false;
            }
            if(target[i] == 'L') l--;
            if(start[i] == 'R') r++;
            if(target[i] == 'R'){
                r--;
                if(l != 0) return false;
            }
            if(l > 0 || r < 0) return false;
        }
        if(l != 0 || r != 0) return false;
        return true;
    }
};
class P_2554 {
    /*You are given an integer array banned and two integers n and maxSum. You are choosing some number of integers following the below rules:
    The chosen integers have to be in the range [1, n].
    Each integer can be chosen at most once.
    The chosen integers should not be in the array banned.
    The sum of the chosen integers should not exceed maxSum. Return the maximum number of integers you can choose following the mentioned rules.
    Example 1:  Input: banned = [1,6,5], n = 5, maxSum = 6
                Output: 2
                Explanation: You can choose the integers 2 and 4.2 and 4 are from the range [1, 5], both did not appear in banned, and their sum is 6, which did not exceed maxSum.

    Example 2:  Input: banned = [1,2,3,4,5,6,7], n = 8, maxSum = 1
                Output: 0
                Explanation: You cannot choose any integer while following the mentioned conditions. */
public:
    int maxCount(vector<int> &banned, int n, int maxSum){
        int ans = 0;
        unordered_set<int> s (banned.begin(),banned.end());
        int sum = 0;
        for(int i = 1;i <= n;i++){
            if(s.find(i) != s.end()) continue;
            else if(sum  + i > maxSum){
                break;
            }
            sum += i;
            ans++;
        }
        return ans;
    }
};
class P_1760 {
    /*You are given an integer array nums where the ith bag contains nums[i] balls. You are also given an integer maxOperations.You can perform the following operation at most maxOperations times:

    Take any bag of balls and divide it into two new bags with a positive number of balls.For example, a bag of 5 balls can become two new bags of 1 and 4 balls, or two new bags of 2 and 3 balls.
    Your penalty is the maximum number of balls in a bag. You want to minimize your penalty after the operations.
    Return the minimum possible penalty after performing the operations.
    Example 1:  Input: nums = [9], maxOperations = 2
                Output: 3
    Explanation:    - Divide the bag with 9 balls into two bags of sizes 6 and 3. [9] -> [6,3].
                    - Divide the bag with 6 balls into two bags of sizes 3 and 3. [6,3] -> [3,3,3].
                    The bag with the most number of balls has 3 balls, so your penalty is 3 and you should return 3.*/
private:
    bool is_possible(int mx_balls_in_bag,vector<int> &ar,int mxops){
        int total_ops = 0;
        for(int i:ar){
            int ops = ceil(i/(double)mx_balls_in_bag) - 1;
            total_ops += ops;

            if(total_ops > mxops) return false;
        }
        return true;
    }
public:
    int minimumSize(vector<int> &nums, int maxOperations){
        int l = 1,r = 0;
        for(int i:nums) r = max(r,i);
        while(l < r){
            int mid = (l + r)/2;
            if(is_possible(mid,nums,maxOperations)) r = mid;
            else l = mid + 1;
        }
        return l;
    }
};
class P_2054 {
    /*You are given a 0-indexed 2D integer array of events where events[i] = [startTimei, endTimei, valuei]. The ith event starts at startTimei and ends at endTimei, and if you attend this event, you will receive a value of valuei. You can choose at most two non-overlapping events to attend such that the sum of their values is maximized.
    Return this maximum sum.Note that the start time and end time is inclusive: that is, you cannot attend two events where one of them starts and the other ends at the same time. More specifically, if you attend an event with end time t, the next event must start at or after t + 1.
    Example 1:  Input: events = [[1,3,2],[4,5,2],[2,4,3]]
                Output: 4
                Explanation: Choose the green events, 0 and 1 for a sum of 2 + 2 = 4.*/
public:
    int maxTwoEvents(vector<vector<int>> &events){
        vector<array<int,3>> times;
        for(auto &i:events){
            // 1 -> start time, 0 -> end time
            times.push_back({i[0],1,i[2]}); //start,1,val
            times.push_back({i[1]+1,0,i[2]}); //end + 1,0,val
        }
        sort(begin(times),end(times));
        int ans = 0,mxval = 0;
        for(auto &i:times){
            //if cur time is start time,find max sum of max end time till now
            if(i[1]) ans = max(ans,i[2] + mxval);
            else mxval = max(mxval,i[2]);
        }
        return ans;
    }
};
class P_3152 {
    /*An array is considered special if every pair of its adjacent elements contains two numbers with different parity.You are given an array of integer nums and a 2D integer matrix queries, where for queries[i] = [fromi, toi] your task is to check that
    subarray nums[fromi..toi] is special or not.
    Return an array of booleans answer such that answer[i] is true if nums[fromi..toi] is special.
    Example 1:  Input: nums = [3,4,1,2,6], queries = [[0,4]]
                Output: [false]
    Explanation:The subarray is [3,4,1,2,6]. 2 and 6 are both even.*/
public:
    vector<bool> isArraySpecial(vector<int> &nums, vector<vector<int>> &queries){
        int n = nums.size();
        vector<int> parityMismatch(n, 0);
        for (int i = 1; i < n; i++)
        {
            parityMismatch[i] = parityMismatch[i - 1] +
                                ((nums[i] % 2 == nums[i - 1] % 2) ? 1 : 0);
        }

        vector<bool> ans(queries.size(), true);

        for (int i = 0; i < queries.size(); i++)
        {
            int s = queries[i][0], e = queries[i][1];
            if (parityMismatch[e] - parityMismatch[s] > 0)
            {
                ans[i] = false;
            }
        }

        return ans;
    }
};
class P_2981 {
    /*You are given a string s that consists of lowercase English letters.A string is called special if it is made up of only a single character. For example, the string "abc" is not special, whereas the strings "ddd", "zz", and "f" are special.
    Return the length of the longest special substring of s which occurs at least thrice, or -1 if no special substring occurs at least thrice.A substring is a contiguous non-empty sequence of characters within a string.
    Example 1:  Input: s = "aaaa"
                Output: 2
                Explanation: The longest special substring which occurs thrice is "aa": substrings "aaaa", "aaaa", and "aaaa".It can be shown that the maximum length achievable is 2.*/
public:
    int maximumLength(string s){
        map<string,int> mp;
        for(int st = 0;st < s.length();st++){
            string cur;
            for(int en = st;en < s.length();en++){
                if(cur.empty() || cur.back() == s[en]){
                    cur.push_back(s[en]);
                    mp[cur]++;
                }else break;
            }
        }
        int ans = 0;
        for(auto &i:mp){
            string tmp = i.first;
            if(i.second >= 3 && tmp.length() > ans) ans = tmp.length();
        }
        return ans == 0?-1:ans;
    }
};
class P_2779 {
    /*You are given a 0-indexed array nums and a non-negative integer k.In one operation, you can do the following:Choose an index i that hasn't been chosen before from the range [0, nums.length - 1].
    Replace nums[i] with any integer from the range [nums[i] - k, nums[i] + k].The beauty of the array is the length of the longest subsequence consisting of equal elements.
    Return the maximum possible beauty of the array nums after applying the operation any number of times.Note that you can apply the operation to each index only once.
    A subsequence of an array is a new array generated from the original array by deleting some elements (possibly none) without changing the order of the remaining elements.

    Example 1:  Input: nums = [4,6,1,2], k = 2  
                Output: 3
    Explanation: In this example, we apply the following operations:
    - Choose index 1, replace it with 4 (from range [4,8]), nums = [4,4,1,2].
    - Choose index 3, replace it with 4 (from range [0,4]), nums = [4,4,1,4].
    After the applied operations, the beauty of the array nums is 3 (subsequence consisting of indices 0, 1, and 3).It can be proven that 3 is the maximum possible length we can achieve*/
public:
    int maximumBeauty(vector<int> &nums, int k){
        sort(begin(nums),end(nums));
        int len = 0,i = 0,j = 0,n = nums.size();
        for(int j = 0;j < n;j++){
            while(nums[j] - nums[i] > 2*k){
                i++;
            }
            len = max(len,j-i+1);
        }
        return len;
    }
};
class P_2558 {
    /*You are given an integer array gifts denoting the number of gifts in various piles. Every second, you do the following:Choose the pile with the maximum number of gifts.
    If there is more than one pile with the maximum number of gifts, choose any.Leave behind the floor of the square root of the number of gifts in the pile. Take the rest of the gifts.
    Return the number of gifts remaining after k seconds.
    Example 1:  Input: gifts = [25,64,9,4,100], k = 4
                Output: 29
    Explanation:The gifts are taken in the following way:
                - In the first second, the last pile is chosen and 10 gifts are left behind.
                - Then the second pile is chosen and 8 gifts are left behind.
                - After that the first pile is chosen and 5 gifts are left behind.
                - Finally, the last pile is chosen again and 3 gifts are left behind.
                The final remaining gifts are [5,8,9,4,3], so the total number of gifts remaining is 29*/
public:
    long long pickGifts(vector<int> &gifts, int k){
        int n = gifts.size();
        while (k--){
            sort(gifts.begin(), gifts.end());
            int mx = gifts[n - 1];
            gifts[n - 1] = sqrt(mx);
        }

        long long sum = 0;
        for (auto &i : gifts)
            sum += i;
        return sum;
    }
};
class P_2593 {
//Asked in ZSCALER
/*You are given an array nums consisting of positive integers.

Starting with score = 0, apply the following algorithm:

    Choose the smallest integer of the array that is not marked. If there is a tie, choose the one with the smallest index.
    Add the value of the chosen integer to score.
    Mark the chosen element and its two adjacent elements if they exist.
    Repeat until all the array elements are marked.

Return the score you get after applying the above algorithm.



Example 1:

Input: nums = [2,1,3,4,5,2]
Output: 7
Explanation: We mark the elements as follows:
- 1 is the smallest unmarked element, so we mark it and its two adjacent elements: [2,1,3,4,5,2].
- 2 is the smallest unmarked element, so we mark it and its left adjacent element: [2,1,3,4,5,2].
- 4 is the only remaining unmarked element, so we mark it: [2,1,3,4,5,2].
Our score is 1 + 2 + 4 = 7.
*/
public:
    /*//Runs but TLE
    int smallest(vector<int>&nums){
        int s = INT_MAX,idx = -1;
        for(int i = 0;i < nums.size();i++){
            if(nums[i] < s && nums[i] != 0){
                idx = i;
                s = nums[i];
            }
        }
        return idx;
    }
    long long findScore(vector<int> &nums){
        long long score = 0;
        for(;;){
            int idx = smallest(nums);
            if(idx == -1) break;

            score += nums[idx];
            nums[idx] = 0;

            if(idx > 0) nums[idx - 1] = 0;
            if(idx < nums.size()-1) nums[idx + 1] = 0;
        }
        return score;
    }*/
   long long findScore(vector<int>& nums){
       long long score = 0;
       vector<pair<int, int>> arr(nums.size());
       for (int i = 0; i < nums.size(); i++)
           arr[i] = make_pair(nums[i], i);
       sort(arr.begin(), arr.end());
       vector<bool> marked(nums.size(), false);
       for (int i = 0; i < nums.size(); i++)
       {
           int num = arr[i].first, idx = arr[i].second;
           if (!marked[idx])
           {
               score += num;
               marked[idx] = true;

               if (idx - 1 >= 0)
                   marked[idx - 1] = true;
               if (idx + 1 < nums.size())
                   marked[idx + 1] = true;
           }
       }
       return score;
   }
};
class P_2762 {
    /*You are given a 0-indexed integer array nums. A subarray of nums is called continuous if:
    Let i, i + 1, ..., j be the indices in the subarray. Then, for each pair of indices i <= i1, i2 <= j, 0 <= |nums[i1] - nums[i2]| <= 2.
    Return the total number of continuous subarrays.
    A subarray is a contiguous non-empty sequence of elements within an array.
    Example 1:Input: nums = [5,4,2,4]
            Output: 8
    Explanation:Continuous subarray of size 1: [5], [4], [2], [4].
                Continuous subarray of size 2: [5,4], [4,2], [2,4].
                Continuous subarray of size 3: [4,2,4].
                Thereare no subarrys of size 4.
    Total continuous subarrays = 4 + 3 + 1 = 8.It can be shown that there are no more continuous subarrays.*/
public:
    long long continuousSubarrays(vector<int> &nums){
        map<int,int> mp;
        int l = 0,r = 0;
        long long count = 0;
        while(r < nums.size()){
            mp[nums[r]]++;
            //nums[i] - nums[j] (difference between largest and smallest shd be >= 2)
            while(mp.rbegin()->first - mp.begin()->first > 2){
                mp[nums[l]]--;
                if(mp[nums[l]] == 0) mp.erase(nums[l]);
                l++;
            }
            count += r - l + 1;
            r++;
        }
        return count;
    }
};
class P_1792 {
    /*There is a school that has classes of students and each class will be having a final exam. You are given a 2D integer array classes, where classes[i] = [passi, totali]. You know beforehand that in the ith class, there are totali total students, but only passi number of students will pass the exam.
    You are also given an integer extraStudents. There are another extraStudents brilliant students that are guaranteed to pass the exam of any class they are assigned to. You want to assign each of the extraStudents students to a class in a way that maximizes the average pass ratio across all the classes.
    The pass ratio of a class is equal to the number of students of the class that will pass the exam divided by the total number of students of the class. The average pass ratio is the sum of pass ratios of all the classes divided by the number of the classes.
    Return the maximum possible average pass ratio after assigning the extraStudents students. Answers within 10-5 of the actual answer will be accepted.
    Example 1:  Input: classes = [[1,2],[3,5],[2,2]], extraStudents = 2
                Output: 0.78333
    Explanation: You can assign the two extra students to the first class. The average pass ratio will be equal to (3/4 + 3/5 + 2/2) / 3 = 0.78333.

    Example 2:Input: classes = [[2,4],[3,9],[4,5],[2,10]], extraStudents = 4
            Output: 0.53485*/
public:
   /* double maxAverageRatio(vector<vector<int>> &classes, int extraStudents){
        auto cal_gain = [](int passes,int total_stud){
            return (double)(passes + 1) / (total_stud + 1) - (double)passes / total_stud;
        };

        priority_queue<pair<double,pair<int,int>>> mxhp; // maxHeap
        for(auto &single_class:classes){
            mxhp.push({cal_gain(single_class[0],single_class[1]),{single_class[0],single_class[1]}});
        }
        while(extraStudents--){
            auto &[currentGain, classInfo] = mxhp.top();
            mxhp.pop();
            int passes = classInfo.first;
            int totalStudents = classInfo.second;
            mxhp.push({cal_gain(passes + 1, totalStudents + 1),{passes + 1, totalStudents + 1}});
        }

        // Calculate the final average pass ratio
        double totalPassRatio = 0;
        while (!mxhp.empty()){
            auto &[i, classInfo] = mxhp.top();
            mxhp.pop();
            totalPassRatio += (double)classInfo.first / classInfo.second;
        }

        return totalPassRatio / classes.size();
    }*/
};
class P_3264 {
    /*You are given an integer array nums, an integer k, and an integer multiplier.You need to perform k operations on nums. In each operation:
    Find the minimum value x in nums. If there are multiple occurrences of the minimum value, select the one that appears first.
    Replace the selected minimum value x with x * multiplier.
    Return an integer array denoting the final state of nums after performing all k operations.
    Input: nums = [2,1,3,5,6], k = 5, multiplier = 2
    Output: [8,4,6,5,6]*/
    public:
        vector<int> getFinalState(vector<int> &nums, int k, int multiplier){
            //BUILT-IN
            while(k--){
                int min_idx = distance(nums.begin(),min_element(nums.begin(),nums.end()));
                nums[min_idx] = nums[min_idx] * multiplier;
            }
            return nums;
            /*  OPTIMISED
            vector<pair<int,int>> heap;
        for(int i = 0;i<nums.size();i++){
            heap.push_back({nums[i],i});
        }
        make_heap(heap.begin(),heap.end(),greater<>());
        while(k--){
            pop_heap(heap.begin(),heap.end(),greater<>());
            auto [val,i] = heap.back();
            heap.pop_back();

            nums[i] *= multiplier;
            heap.push_back({nums[i],i});
            push_heap(heap.begin(),heap.end(),greater<>());
        }
        return nums;*/
        }
};
class P_2182 {
    /*You are given a string s and an integer repeatLimit. Construct a new string repeatLimitedString using the characters of s such that no letter appears more than repeatLimit times in a row. You do not have to use all characters from s.
    Return the lexicographically largest repeatLimitedString possible.A string a is lexicographically larger than a string b if in the first position where a and b differ, string a has a letter that appears later in the alphabet than the corresponding letter in b. If the first min(a.length, b.length) characters do not differ, then the longer string is the lexicographically larger one.
    Example 1:Input: s = "cczazcc", repeatLimit = 3
              Output: "zzcccac"
    Explanation: We use all of the characters from s to construct the repeatLimitedString "zzcccac".
    The letter 'a' appears at most 1 time in a row.
    The letter 'c' appears at most 3 times in a row.
    The letter 'z' appears at most 2 times in a row.
    Hence, no letter appears more than repeatLimit times in a row and the string is a valid repeatLimitedString.
    The string is the lexicographically largest repeatLimitedString possible so we return "zzcccac".
    Note that the string "zzcccca" is lexicographically larger but the letter 'c' appears more than 3 times in a row, so it is not a valid repeatLimitedString.*/
public:
    string repeatLimitedString(string s, int repeatLimit){
        string ans;
        vector<int> freq(26,0);
        for(char c:s) freq[c - 'a']++;
        int cur_char_idx = 25;
        while(cur_char_idx >= 0){
            if(freq[cur_char_idx] == 0){ cur_char_idx--; continue; }

            int repeat = min(freq[cur_char_idx],repeatLimit);
            ans.append(repeat,'a' + cur_char_idx);
            freq[cur_char_idx] -= repeat;

            if(freq[cur_char_idx] > 0) {
                int smaller_idx = cur_char_idx - 1;
                while(smaller_idx >= 0 && freq[smaller_idx] == 0){
                    smaller_idx--;
                }
                if(smaller_idx < 0) break;
                ans.push_back('a' + smaller_idx);
                freq[smaller_idx]--;
            }
        }
        return ans;
    }
};
class P_1475 {
public:
    vector<int> finalPrices(vector<int> &prices){
        /* monotonic stack
        vector<int> ans = prices;
        stack<int> s;
        for(int i = 0;i < prices.size();i++){
            while(!s.empty() && prices[s.top()] >= prices[i]){
                ans[s.top()] -= prices[i];
                s.pop();
            }
            s.push(i);
        }
        return ans;*/
        vector<int> ans(prices.size(),-1);
        for(int i = 0;i<prices.size();i++){
            int cur = prices[i];
            for(int j = i+1;j < prices.size();j++){
                if(prices[j] <= cur){
                    ans[i] = cur - prices[j];
                    break;
                }
            }
            if(ans[i] == -1) ans[i] = cur;
        }
        return ans;
    }
};
class P_769 {
    /*You are given an integer array arr of length n that represents a permutation of the integers in the range [0, n - 1].We split arr into some number of chunks (i.e., partitions), and individually sort each chunk. After concatenating them, the result should equal the sorted array.
    Return the largest number of chunks we can make to sort the array.
    Example 1:
    Input: arr = [4,3,2,1,0]
    Output: 1
    Explanation:Splitting into two or more chunks will not return the required result.
For example, splitting into [4, 3], [2, 1, 0] will result in [3, 4, 0, 1, 2], which isn't sorted.
   Example 2:Input: arr = [1,0,2,3,4]
   Output: 4
    Explanation:We can split into two chunks, such as [1, 0], [2, 3, 4].However, splitting into [1, 0], [2], [3], [4] is the highest number of chunks possible.*/
public:
    int maxChunksToSorted(vector<int> &arr){
        int chunks = 0,maxele = 0;
        for(int i = 0;i < arr.size();i++){
            maxele = max(maxele,arr[i]);
            if(maxele == i) chunks++;
        }
        return chunks;
    }
};
class P_2415 {
    /*Given the root of a perfect binary tree, reverse the node values at each odd level of the tree.For example, suppose the node values at level 3 are [2,1,3,4,7,11,29,18], then it should become [18,29,11,7,4,3,1,2].
    Return the root of the reversed tree.A binary tree is perfect if all parent nodes have two children and all leaves are on the same level.
    The level of a node is the number of edges along the path between it and the root node.
    Example 1:Input: root = [2,3,5,8,13,21,34]
    Output: [2,5,3,8,13,21,34]
    Explanation:
    The tree has only one odd level.
    The nodes at level 1 are 3, 5 respectively, which are reversed and become 5, 3.*/
    public:
        /* BFS
        TreeNode *reverseOddLevels(TreeNode *root){
            if (!root){
                return nullptr; // Return null if the tree is empty.
            }

            queue<TreeNode *> q;
            q.push(root); // Start BFS with the root node.
            int level = 0;

            while (!q.empty()){
                int size = q.size();
                vector<TreeNode *> currentLevelNodes;

                // Process all nodes at the current level.
                for (int i = 0; i < size; i++){
                    TreeNode *node = q.front();
                    q.pop();
                    currentLevelNodes.push_back(node);

                    if (node->left)
                        q.push(node->left);
                    if (node->right)
                        q.push(node->right);
                }

                // Reverse node values if the current level is odd.
                if (level % 2 == 1){
                    int left = 0, right = currentLevelNodes.size() - 1;
                    while (left < right){
                        swap(currentLevelNodes[left]->val, currentLevelNodes[right]->val);
                        left++;
                        right--;
                    }
                }
                level++;
            }

            return root; // Return the modified tree root.
        }*/
        /* private:
     void DFS(TreeNode* lc,TreeNode* rc,int lvl) { //left & right child & level
         if(!lc || !rc) return;

         if(lvl % 2 == 0){ // if even level, swap children vals
             //int tmp =  lc->val;
             //lc->val = rc->val;
             //rc->val = tmp; OR use built-in
        swap(lc->val, rc->val);
        } 
        DFS(lc->left, rc->right, lvl + 1);
        DFS(lc->right, rc->left, lvl + 1);
    }

public:
TreeNode *reverseOddLevels(TreeNode *root)
{
    DFS(root->left, root->right, 0);
    return root;
}*/
};
class P_2872 {
    /*There is an undirected tree with n nodes labeled from 0 to n - 1. You are given the integer n and a 2D integer array edges of length n - 1, where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.
You are also given a 0-indexed integer array values of length n, where values[i] is the value associated with the ith node, and an integer k.
A valid split of the tree is obtained by removing any set of edges, possibly empty, from the tree such that the resulting components all have values that are divisible by k, where the value of a connected component is the sum of the values of its nodes.
Return the maximum number of components in any valid split
Input: n = 5, edges = [[0,2],[1,2],[1,3],[2,4]], values = [1,8,1,4,4], k = 6
Output: 2
Explanation: We remove the edge connecting node 1 with 2. The resulting split is valid because:
- The value of the component containing nodes 1 and 3 is values[1] + values[3] = 12.
- The value of the component containing nodes 0, 2, and 4 is values[0] + values[2] + values[4] = 6.
It can be shown that no other valid split has more than 2 connected components.*/
public:
    int maxKDivisibleComponents(int n, vector<vector<int>> &edges, vector<int> &values, int k){
        if(n < 2) return 1;
        int component = 0;
        unordered_map<int,unordered_set<int>> graph;
        // step - 1 : Build Graph
        for(auto &i:edges){
            int node1 = i[0],node2 = i[1];
            graph[node1].insert(node2);
            graph[node2].insert(node1);
        }  
        //Convert values to long long to prevent overflow
        vector<long long> long_val (values.begin(),values.end());
        //step - 2: Initialise the BFS Q with only leaf nodes
        queue<int> q;
        for(auto &i:graph){ //i =  [node,neighbors]
            if(i.second.size() == 1)
                q.push(i.first);
        }

        //step - 3: Process Nodes in BFS Order
        while(!q.empty()){
            int cur_node = q.front();
            q.pop();

            int neighbor_node = -1;
            if(!graph[cur_node].empty()){
                neighbor_node  = *graph[cur_node].begin();
            }
            if(neighbor_node >= 0){
                //remove edge between current and neighbor
                graph[neighbor_node].erase(cur_node);
                graph[cur_node].erase(neighbor_node);
            }

            //Check divisibility of the current node's value
            if(long_val[cur_node] % k == 0){
                component++;
            }else if(neighbor_node >= 0){
                // Add current node's value to the neighbor
                long_val[neighbor_node] += long_val[cur_node];
            }
            //if node becomes leaf then add to queue
            if(neighbor_node >= 0 && graph[neighbor_node].size() == 1){
                q.push(neighbor_node);
            }
        }
        return component;
    }
};
class P_2940 {
public:
    /*You are given a 0-indexed array heights of positive integers, where heights[i] represents the height of the ith building.
    If a person is in building i, they can move to any other building j if and only if i < j and heights[i] < heights[j].
    You are also given another array queries where queries[i] = [ai, bi]. On the ith query, Alice is in building ai while Bob is in building bi.
    Return an array ans where ans[i] is the index of the leftmost building where Alice and Bob can meet on the ith query. If Alice and Bob cannot move to a common building on query i, set ans[i] to -1.
    Example 1:
    Input: heights = [6,4,8,5,2,7], queries = [[0,1],[0,3],[2,4],[3,4],[2,2]]
    Output: [2,5,-1,5,2]
    Explanation: In the first query, Alice and Bob can move to building 2 since heights[0] < heights[2] and heights[1] < heights[2].
    In the second query, Alice and Bob can move to building 5 since heights[0] < heights[5] and heights[3] < heights[5].
    In the third query, Alice cannot meet Bob since Alice cannot move to any other building.
    In the fourth query, Alice and Bob can move to building 5 since heights[3] < heights[5] and heights[4] < heights[5].
    In the fifth query, Alice and Bob are already in the same building.
    For ans[i] != -1, It can be shown that ans[i] is the leftmost building where Alice and Bob can meet.
    For ans[i] == -1, It can be shown that there is no building where Alice and Bob can meet.

    Example 2:
    Input: heights = [5,3,8,2,6,1,4,6], queries = [[0,7],[3,5],[5,2],[3,0],[1,6]]
    Output: [7,6,-1,4,6]
    Explanation: In the first query, Alice can directly move to Bob's building since heights[0] < heights[7].
    In the second query, Alice and Bob can move to building 6 since heights[3] < heights[6] and heights[5] < heights[6].
    In the third query, Alice cannot meet Bob since Bob cannot move to any other building.
    In the fourth query, Alice and Bob can move to building 4 since heights[3] < heights[4] and heights[0] < heights[4].
    In the fifth query, Alice can directly move to Bob's building since heights[1] < heights[6].
    For ans[i] != -1, It can be shown that ans[i] is the leftmost building where Alice and Bob can meet.
    For ans[i] == -1, It can be shown that there is no building where Alice and Bob can meet.

    */
    private:
        int bin_search(int height,vector<pair<int,int>> &stk){
            int l = 0,r = stk.size()-1,ans = -1;
            while(l <= r){
                int mid = (l+r)/2;
                if(stk[mid].first > height){
                    ans = max(ans,mid);
                    l = mid + 1;
                }else
                    r = mid - 1;
            }
            return ans;
        }
    vector<int> leftmostBuildingQueries(vector<int> &heights, vector<vector<int>> &queries){
        vector<pair<int,int>> monostack;
        vector<vector<pair<int, int>>> new_queries(heights.size()); //Sorted Queries
        vector<int> res(queries.size(),-1);

        for(int i = 0;i < queries.size();i++){
            int a = queries[i][0],b = queries[i][1];
            if(a > b) swap(a,b);
            if(heights[b] > heights[a] || a == b)
                res[i] = b;
            else
                new_queries[b].push_back({heights[a],i});
        }
        for(int i = heights.size() - 1;i >= 0;i--){
            int stacksiz = monostack.size();
            for(auto &[a,b]: new_queries[i]){
                int pos = bin_search(a,monostack);
                if(pos < stacksiz && pos >= 0)
                    res[b] = monostack[pos].second;
            }
            while(!monostack.empty() && monostack.back().first <= heights[i])
                monostack.pop_back();
            monostack.push_back({heights[i],i});
        }
        return res;
    }
};
class P_2471 {
    /*You are given the root of a binary tree with unique values.In one operation, you can choose any two nodes at the same level and swap their values.
    Return the minimum number of operations needed to make the values at each level sorted in a strictly increasing order.
    The level of a node is the number of edges along the path between it and the root node.
Example 1:

Input: root = [1,4,3,7,6,8,5,null,null,null,null,9,null,10]
Output: 3
Explanation:
- Swap 4 and 3. The 2nd level becomes [3,4].
- Swap 7 and 5. The 3rd level becomes [5,6,8,7].
- Swap 8 and 7. The 3rd level becomes [5,6,7,8].
We used 3 operations so return 3.
It can be proven that 3 is the minimum number of operations needed.*/
/*private:
    int min_swaps(vector<int> &arr){
        int swaps = 0;
        vector<int> target(arr.begin(), arr.end());
        sort(begin(target), end(target));
        unordered_map<int, int> pos;
        for (int i = 0; i < arr.size(); i++){
            pos[arr[i]] = i;
        }
        for (int i = 0; i < arr.size(); i++){
            if (arr[i] != target[i]){
                swaps++;

                int curpos = pos[target[i]];
                pos[arr[i]] = curpos;
                swap(arr[curpos], arr[i]);
            }
        }
        return swaps;
    }

public:
    int minimumOperations(TreeNode *root){
        int total_swaps = 0;
        if (!root)
            return 0;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()){
            int siz = q.size();
            vector<int> lvl(siz, 0);
            for (int i = 0; i < siz; i++){
                TreeNode *cur = q.front();
                q.pop();
                lvl[i] = cur->val;
                if (cur->left)
                    q.push(cur->left);
                if (cur->right)
                    q.push(cur->right);
            }
            total_swaps += min_swaps(lvl);
        }
        return total_swaps;
    }*/
};
class P_515 {
    //Given the root of a binary tree, return an array of the largest value in each row of the tree(0 - indexed).
    /*
    public : 
    vector<int> largestValues(TreeNode * root){
        vector<int> ans;
        if (!root)
            return ans;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()){
            int s = q.size(), curmx = INT_MIN;
            for (int i = 0; i < s; i++){
                TreeNode *cur = q.front();
                q.pop();
                curmx = max(curmx, cur->val);
                if (cur->left)
                    q.push(cur->left);
                if (cur->right)
                    q.push(cur->right);
            }
            ans.push_back(curmx);
        }
        return ans;
    }*/
};
class P_494 {
    /*You are given an integer array nums and an integer target.You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
    For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
    Return the number of different expressions that you can build, which evaluates to target.

Example 1:Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

Example 2:Input: nums = [1], target = 1
Output: 1
*/
public:
    int solve(vector<int> nums, int idx, int cursum, int target, vector<vector<int>> &dp, int sum)
    {
        if (idx == nums.size())
        {
            if (cursum == target)
                return 1;
            else
                return 0;
        }
        if (dp[idx][cursum + sum] != -1)
            return dp[idx][cursum + sum];

        int plus = solve(nums, idx + 1, cursum + nums[idx], target, dp, sum);
        int minus = solve(nums, idx + 1, cursum - nums[idx], target, dp, sum);

        return dp[idx][cursum + sum] = (plus + minus);
    }
    int findTargetSumWays(vector<int> &nums, int target)
    {
        int n = nums.size();
        int total_sum = accumulate(nums.begin(), nums.end(), 0);
        vector<vector<int>> dp(n + 1, vector<int>(2 * total_sum + 1, -1));
        return solve(nums, 0, 0, target, dp, total_sum);
    }
    int solve(vector<int> nums, int idx, int cursum, int target, vector<vector<int>> &dp, int sum)
    {
        if (idx == nums.size())
        {
            if (cursum == target)
                return 1;
            else
                return 0;
        }
        if (dp[idx][cursum + sum] != -1)
            return dp[idx][cursum + sum];

        int plus = solve(nums, idx + 1, cursum + nums[idx], target, dp, sum);
        int minus = solve(nums, idx + 1, cursum - nums[idx], target, dp, sum);

        return dp[idx][cursum + sum] = (plus + minus);
    }
    int findTargetSumWays(vector<int> &nums, int target)
    {
        int n = nums.size();
        int total_sum = accumulate(nums.begin(), nums.end(), 0);
        vector<vector<int>> dp(n + 1, vector<int>(2 * total_sum + 1, -1));
        return solve(nums, 0, 0, target, dp, total_sum);
    }
    int solve(vector<int> nums, int idx, int cursum, int target, unordered_map<string, int> &mp){
        if (idx == nums.size()){
            if (cursum == target)
                return 1;
            else
                return 0;
        }
        string key = to_string(idx) + "," + to_string(cursum);
        if (mp.count(key))
            return mp[key];

        int plus = solve(nums, idx + 1, cursum + nums[idx], target, mp);
        int minus = solve(nums, idx + 1, cursum - nums[idx], target, mp);

        return mp[key] = (plus + minus);
    }
    int findTargetSumWays(vector<int> &nums, int target){
        unordered_map<string, int> mp;
        return solve(nums, 0, 0, target, mp);
    }
};
class P_1014 {
    /*You are given an integer array values where values[i] represents the value of the ith sightseeing spot. Two sightseeing spots i and j have a distance j - i between them.
    The score of a pair (i < j) of sightseeing spots is values[i] + values[j] + i - j: the sum of the values of the sightseeing spots, minus the distance between them.
    Return the maximum score of a pair of sightseeing spots.
    Example 1:  
    Input: values = [8,1,5,2,6]
    Output: 11
    Explanation: i = 0, j = 2, values[i] + values[j] + i - j = 8 + 5 + 0 - 2 = 11

    Example 2:
    Input: values = [1,2]
    Output: 2*/
public:
    int maxScoreSightseeingPair(vector<int> &values){
        int n = values.size(), mx_score = 0;
        vector<int> mxleft_score(n);
        mxleft_score[0] = values[0];
        for (int i = 1; i < n; i++){
            int cur_right_score = values[i] - i;
            mx_score = max(mx_score, mxleft_score[i - 1] + cur_right_score);
            int cur_left_score = values[i] + i;
            mxleft_score[i] = max(mxleft_score[i - 1], cur_left_score);
        }
        return mx_score;
    }
};
class P_689 {
    /*  Maximum Sum of 3 Non-Overlapping Subarrays
    Given an integer array nums and an integer k, find three non-overlapping subarrays of length k with maximum sum and return them.
    Return the result as a list of indices representing the starting position of each interval (0-indexed). If there are multiple answers, return the lexicographically smallest one.
    Example 1:
    Input: nums = [1,2,1,2,6,7,5,1], k = 2
    Output: [0,3,5]
    Explanation: Subarrays [1, 2], [2, 6], [7, 5] correspond to the starting indices [0, 3, 5].
    We could have also taken [2, 1], but an answer of [1, 3, 5] would be lexicographically larger.

    Example 2:

    Input: nums = [1,2,1,2,1,2,1,2,1], k = 2
    Output: [0,2,4]
    */
private:
    int dp(vector<int> &sums, int k, int idx, int rem, vector<vector<int>> &memo){
        if (rem == 0)
            return 0;
        if (idx >= sums.size()){
            return rem > 0 ? INT_MIN : 0;
        }
        if (memo[idx][rem] != -1){
            return memo[idx][rem];
        }
        int withcur = sums[idx] + dp(sums, k, idx + k, rem - 1, memo);
        int skipcur = dp(sums, k, idx + 1, rem, memo);

        memo[idx][rem] = max(withcur, skipcur);
        return memo[idx][rem];
    }
    void dfs(vector<int> &sums, int k, int idx, int rem, vector<vector<int>> &memo, vector<int> &indices){
        if (rem == 0)
            return;
        if (idx >= sums.size())
            return;

        int withcur = sums[idx] + dp(sums, k, idx + k, rem - 1, memo);
        int skipcur = dp(sums, k, idx + 1, rem, memo);

        if (withcur >= skipcur){
            indices.push_back(idx);
            dfs(sums, k, idx + k, rem - 1, memo, indices);
        }
        else{
            dfs(sums, k, idx + 1, rem, memo, indices);
        }
    }

public:
    vector<int> maxSumOfThreeSubarrays(vector<int> &nums, int k){
        int n = nums.size() - k + 1;
        vector<int> sums(n);
        int winsum = 0;
        for (int i = 0; i < k; i++){
            winsum += nums[i];
        }
        sums[0] = winsum;
        for (int i = k; i < nums.size(); i++){
            winsum = winsum - nums[i - k] + nums[i];
            sums[i - k + 1] = winsum;
        }
        vector<vector<int>> memo(n, vector<int>(4, -1));
        vector<int> ind;

        dp(sums, k, 0, 3, memo);
        dfs(sums, k, 0, 3, memo, ind);
        return ind;
    }
};
class P_1639 {
    /* Number of Ways to Form a Target String Given a Dictionary
    You are given a list of strings of the same length words and a string target.Your task is to form target using the given words under the following rules:

    target should be formed from left to right.
    To form the ith character (0-indexed) of target, you can choose the kth character of the jth string in words if target[i] = words[j][k].
    Once you use the kth character of the jth string of words, you can no longer use the xth character of any string in words where x <= k. In other words, all characters to the left of or at index k become unusuable for every string.
    Repeat the process until you form the string target.

Notice that you can use multiple characters from the same string in words provided the conditions above are met.

Return the number of ways to form target from words. Since the answer may be too large, return it modulo 109 + 7.
Example 1:

Input: words = ["acca","bbbb","caca"], target = "aba"
Output: 6
Explanation: There are 6 ways to form target.
"aba" -> index 0 ("acca"), index 1 ("bbbb"), index 3 ("caca")
"aba" -> index 0 ("acca"), index 2 ("bbbb"), index 3 ("caca")
"aba" -> index 0 ("acca"), index 1 ("bbbb"), index 3 ("acca")
"aba" -> index 0 ("acca"), index 2 ("bbbb"), index 3 ("acca")
"aba" -> index 1 ("caca"), index 2 ("bbbb"), index 3 ("acca")
"aba" -> index 1 ("caca"), index 2 ("bbbb"), index 3 ("caca")

Example 2:

Input: words = ["abba","baab"], target = "bab"
Output: 4
Explanation: There are 4 ways to form target.
"bab" -> index 0 ("baab"), index 1 ("baab"), index 2 ("abba")
"bab" -> index 0 ("baab"), index 1 ("baab"), index 3 ("baab")
"bab" -> index 0 ("baab"), index 2 ("baab"), index 3 ("baab")
"bab" -> index 1 ("abba"), index 2 ("baab"), index 3 ("baab")
*/
public:
    int numWays(vector<string> &words, string target){
        int wl = words[0].length(), tl = target.length();
        const int mod = 1e9 + 7;

        vector<vector<int>> freq(wl, vector<int>(26, 0));
        for (const string &w : words){
            for (int j = 0; j < wl; j++){
                freq[j][w[j] - 'a']++;
            }
        }
        vector<vector<long>> dp(wl + 1, vector<long>(tl + 1, 0));
        for (int cur = 0; cur <= wl; cur++){
            dp[cur][0] = 1;
        }
        for (int cur = 1; cur <= wl; cur++){
            for (int ct = 1; ct <= tl; ct++){
                dp[cur][ct] = dp[cur - 1][ct];
                int curpos = target[ct - 1] - 'a';
                dp[cur][ct] += (freq[cur - 1][curpos] * dp[cur - 1][ct - 1]) % mod;
                dp[cur][ct] %= mod;
            }
        }
        return dp[wl][tl];
    }
};
class P_2466 {
    /* Count Ways To Build Good Strings
    Given the integers zero, one, low, and high, we can construct a string by starting with an empty string, and then at each step perform either of the following:

    Append the character '0' zero times.
    Append the character '1' one times.

This can be performed any number of times.

A good string is a string constructed by the above process having a length between low and high (inclusive).

Return the number of different good strings that can be constructed satisfying these properties. Since the answer can be large, return it modulo 109 + 7.
Example 1:

Input: low = 3, high = 3, zero = 1, one = 1
Output: 8
Explanation:
One possible valid good string is "011".
It can be constructed as follows: "" -> "0" -> "01" -> "011".
All binary strings from "000" to "111" are good strings in this example.

Example 2:

Input: low = 2, high = 3, zero = 1, one = 2
Output: 5
Explanation: The good strings are "00", "11", "000", "110", and "011".
*/
public:
    int countGoodStrings(int low, int high, int zero, int one){
        vector<int> dp(high + 1);
        dp[0] = 1;
        int mod = 1e9 + 7;

        for (int end = 1; end <= high; ++end){
            if (end >= zero){
                dp[end] += dp[end - zero];
            }
            if (end >= one){
                dp[end] += dp[end - one];
            }
            dp[end] %= mod;
        }

        int answer = 0;
        for (int i = low; i <= high; ++i){
            answer += dp[i];
            answer %= mod;
        }
        return answer;
    }
};
class P_983 {
    /* Minimum Cost For Tickets
    You have planned some train traveling one year in advance. The days of the year in which you will travel are given as an integer array days. Each day is an integer from 1 to 365.
    Train tickets are sold in three different ways:

    a 1-day pass is sold for costs[0] dollars,
    a 7-day pass is sold for costs[1] dollars, and
    a 30-day pass is sold for costs[2] dollars.

The passes allow that many days of consecutive travel.
For example, if we get a 7-day pass on day 2, then we can travel for 7 days: 2, 3, 4, 5, 6, 7, and 8.
Return the minimum number of dollars you need to travel every day in the given list of days.
Example 1:

Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 1-day pass for costs[0] = $2, which covered day 1.
On day 3, you bought a 7-day pass for costs[1] = $7, which covered days 3, 4, ..., 9.
On day 20, you bought a 1-day pass for costs[0] = $2, which covered day 20.
In total, you spent $11 and covered all the days of your travel.

Example 2:

Input: days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
Output: 17
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 30-day pass for costs[2] = $15 which covered days 1, 2, ..., 30.
On day 31, you bought a 1-day pass for costs[0] = $2 which covered day 31.
In total, you spent $17 and covered all the days of your travel.

*/
public:
    int mincostTickets(vector<int> &days, vector<int> &costs){
        int lastDay = days[days.size() - 1];
        vector<int> dp(lastDay + 1, 0);

        int i = 0;
        for (int day = 1; day <= lastDay; day++){
            if (day < days[i]){
                dp[day] = dp[day - 1];
            }
            else{
                i++;
                dp[day] = min({dp[day - 1] + costs[0],
                               dp[max(0, day - 7)] + costs[1],
                               dp[max(0, day - 30)] + costs[2]});
            }
        }
        return dp[lastDay];
    }
};
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    P_769 s;
    vector<int> a1 = {4, 3, 2, 1, 0}, a2 = {1, 0, 2, 3, 4}, a3 = {10, 1, 1, 6};
    vector<vector<int>> q1 = {{0, 4}}, q2 = {{0, 2}, {2, 3}};
    string s1 = "aaaa", s2 = "abcdef", s3 = "abcaba", s4 = "abcccccdddd";
    cout << s.maxChunksToSorted(a1);
    cout << s.maxChunksToSorted(a2);
    return 0;
}