#include <bits/stdc++.h>
#include "print_arr.h"
#define lli long long int
using namespace std;
class P_2601 {
public:
    bool primeSubOperation(vector<int>& nums) {
        unordered_set<int> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997};
        int i = 0,cur = 1,n = nums.size();
        while(i < n){
            int diff = nums[i] - cur;
            if(diff < 0) return false;
            if(diff == 0 || primes.find(diff) != primes.end()){
                i++;
                cur++;
            }
            else{
                cur++;
            }
        }
        return true;
    }
};
class P_2070 {
public:
    vector<int> maximumBeauty(vector<vector<int>> &items, vector<int> &queries)
    {
        /*Correct but TLE
        vector<int> ans;
        sort(items.begin(),items.end());
        for(int i = 0;i < queries.size();i++){
            int q = queries[i];
            int maxi = 0;
            for(auto &x:items){
                if(x[0] <= q){
                    maxi = max(maxi,x[1]);
                }
                else break;
            }
            ans.push_back(maxi);
        }
        return ans;*/
        //TC = O((M+N)logM) SC = O(Sm) ->m = is,n = qs
        vector<int> ans(queries.size());
        sort(items.begin(), items.end());
        int maxi = items[0][1];
        for(int i = 0;i < items.size();i++){
            maxi = max(maxi,items[i][1]);
            items[i][1] = maxi;
        }

        for(int i = 0;i < queries.size();i++){
            ans[i] = binary_search(items,queries[i]);
        }
        
        //for(auto i:items){cout<<i[0]<<" "<<i[1]<<"\n";}
        return ans;
    }
    int binary_search(vector<vector<int>> &it,int q){
        int l = 0,r = it.size()-1;
        int mxbeauty = 0;
        while(l <= r){
            int mid = (l + r)/2;
            if(it[mid][0] > q){
                r = mid - 1;
            }else{
                mxbeauty = max(mxbeauty,it[mid][1]);
                l = mid + 1;
            }
        }
        return mxbeauty;
    }
};
class P_2563 {
public:
    long long lower_bnd(vector<int>& ar,int val){
        int l = 0,r = ar.size() - 1;
        long long ans = 0;
        while(l < r){
            int sum = ar[l] + ar[r];
            if(sum < val){
                ans += (r - l);
                l++;
            }else{
                r--;
            }
        }
        return ans;
    }
    long long countFairPairs(vector<int> &nums, int lower, int upper)
    {
        sort(nums.begin(),nums.end());
        return lower_bnd(nums,upper+1) - lower_bnd(nums,lower);
    }
};
class P_2064 {
private:
    int help(vector<int>& ar,int rate){
        int totalrate = 0;
        for(int i:ar){
            totalrate += ceil((double)i / (double)rate);
        }
        return totalrate;
    }
public:
    int minimizedMaximum(vector<int> &quantities,int n) {
        //Same as KOKO eating Bananas problem
        int l = 1,r = *max_element(quantities.begin(),quantities.end());
        while(l <= r){
            int mid = l + (r - l)/2;
            long int items = help(quantities,mid);
            if(items <= n){
                r = mid - 1;
            }else{
                l = mid + 1;
            }
        }
        return l;

    }
};
class P_1574 {
public:
    int findLengthOfShortestSubarray(vector<int> &arr){
        int l = 0,r = arr.size()-1;
        while (r > 0 && (arr[r] >= arr[r - 1]))
        {
            r--;
        }
        int ans = r;
        while(l < r && (l == 0 || arr[l-1] <= arr[l])){
            while(r < arr.size() && arr[l] > arr[r]){
                r++;
            }
            ans = min(ans,r - l - 1);
            l++;
        }
        return ans;
    }
};    
class P_3254 {
public:
    bool is_con_sort(vector<int>&arr,int s,int e){
        for(int i = s;i < e;i++){
            if(arr[i] >= arr[i+1] || arr[i+1] != arr[i] + 1)
                return false;
        }
        return true;
    }
    vector<int> resultsArray(vector<int> &nums, int k){
        vector<int> ans;

        for(int i = 0;i < nums.size();i++){
            if ((i + k - 1) >= nums.size()) break;
            if (is_con_sort(nums, i, i + k - 1))
            {
                int m = *max_element(nums.begin() + i, nums.begin() + (i + k));
                ans.push_back(m);
                }else{
                    ans.push_back(-1);
                }
                //if((i+k-1) > nums.size()) break;
        }
        return ans;
    }
};
class P_862  {
public:
    int shortestSubarray(vector<int> &nums, int k){
        /*int ans = INT_MAX,i = 0,j = 0,n = nums.size(),sum = 0;
        while(j < n){
            sum += nums[j];
           // if(sum < k) j++;
            //else {
                while(sum >= k){
                    sum -= nums[i];
                    ans = min(ans, j - i + 1);
                    i++;
            //}
            
            }
            j++;
            
            
        }
        return ans == INT_MAX ? -1 : ans;*/
        int n = nums.size(),subarrlen = INT_MAX;
        vector<long long> prefixsum(n+1,0);
        for(int i = 1;i<=n;i++) prefixsum[i] = prefixsum[i-1] + nums[i-1];

        deque<int> cand_idx;//candidate index
        for(int i = 0;i<=n;i++){
            while(!cand_idx.empty() && prefixsum[i] - prefixsum[cand_idx.front()] >= k){
                subarrlen = min(subarrlen,i - cand_idx.front());
                cand_idx.pop_front();
            }

            while(!cand_idx.empty() && prefixsum[i] <= prefixsum[cand_idx.back()]){
                cand_idx.pop_back();
            }
            cand_idx.push_back(i);
        }
        return subarrlen == INT_MAX?-1:subarrlen;

    }
};
class P_1652 {
public:
    vector<int> decrypt1(vector<int> &code, int k)
    {
        int n = code.size();
        vector<int> ans(n, 0);
        if (k == 0)
            return ans;
        for (int i = 0; i < n; i++)
        {
            if (k > 0)
            {
                for (int j = i + 1; j < i + k + 1; j++)
                    ans[i] += code[j % n];
            }
            else
            {
                for (int j = i - abs(k); j < i; j++)
                {
                    ans[i] += code[(j + n) % n];
                }
            }
        }
        return ans;
    }
    vector<int> decrypt2(vector<int> &code,int k){
        int n = code.size();
        vector<int> ans(n,0);
        if(k == 0) return ans;
        int start = 1,end = k,sum = 0;
        if(k < 0){
            start = n - abs(k),end = n - 1;
        }
        for(int i = start;i <= end;i++) sum += code[i];
        for(int i = 0;i < n;i++){
            ans[i] = sum;
            sum += code[(end+1) % code.size()];
            sum -= code[start % n];
            start++,end++;
        }
        return ans;
    }
};
class P_2461 {
public:
    long long maximumSubarraySum(vector<int> &nums, int k){
        int i = 0,j = 0;
        long long maxsum = 0,cursum = 0;
        unordered_map<int, int> num_idx;
        while(j < nums.size()){
            int cur = nums[j];
            int last_occ = (num_idx.count(cur)?num_idx[cur]:-1);
            while(i <= last_occ || j - i + 1 > k){
                cursum -= nums[i];
                i++;
            }
            num_idx[cur] = j;
            cursum += nums[j];
            if(j-i+1 == k){ maxsum = max(maxsum,cursum); }
            j++;
        }   
        return maxsum;
    }
};
class P_2516 {
    public:
        int takeCharacters(string s, int k){
            int count[3] = {0, 0, 0};
            for (char i : s)
                count[i - 'a']++;
            for (int i = 0; i < 3; i++)
                if (count[i] < k)
                    return -1;
            vector<int> window(3, 0);
            int left = 0, maxwindow = 0, right = 0;
            while (right < s.size())
            {
                window[s[right] - 'a']++;

                while (left <= right && (count[0] - window[0] < k || count[1] - window[1] < k || count[2] - window[2] < k))
                {
                    window[s[left] - 'a']--;
                    left++;
                }
                maxwindow = max(maxwindow, right - left + 1);
                right++;
            }
            return s.size() - maxwindow;
        }
};
class P_2257 {
    public:
    public:
        const int Un = 0, Wall = 1, Guard = 2, Guarded = 3;
        void mark_guard(vector<vector<int>> &grid, int r, int c)
        {
            // UP
            for (int i = r - 1; i >= 0; i--)
            {
                if (grid[i][c] == Wall || grid[i][c] == Guard)
                    break;
                grid[i][c] = Guarded;
            }
            // DOWN
            for (int i = r + 1; i < grid.size(); i++)
            {
                if (grid[i][c] == Wall || grid[i][c] == Guard)
                    break;
                grid[i][c] = Guarded;
            }
            // LEFT
            for (int i = c - 1; i >= 0; i--)
            {
                if (grid[r][i] == Wall || grid[r][i] == Guard)
                    break;
                grid[r][i] = Guarded;
            }
            // RIGHT
            for (int i = c + 1; i < grid[r].size(); i++)
            {
                if (grid[r][i] == Wall || grid[r][i] == Guard)
                    break;
                grid[r][i] = Guarded;
            }
        }
        int countUnguarded(int m, int n, vector<vector<int>> &guards, vector<vector<int>> &walls)
        {

            vector<vector<int>> grid(m, vector<int>(n, Un));
            // print_matrix(grid);
            for (const auto &i : guards)
            {
                grid[i[0]][i[1]] = Guard;
            }
            for (const auto &i : walls)
            {
                grid[i[0]][i[1]] = Wall;
            }
            for (const auto &i : guards)
            {
                mark_guard(grid, i[0], i[1]);
            }

            int ans = 0;
            for (const auto &r : grid)
            {
                for (const auto &c : r)
                {
                    if (c == Un)
                        ans++;
                }
            }
            return ans;
        }
};
class P_1072 {
public:
    int maxEqualRowsAfterFlips(vector<vector<int>> &matrix)
    {
        unordered_map<string, int> mp;
        for (auto &row : matrix)
        {
            string pattern = "";
            for (int c = 0; c < row.size(); c++)
            {
                // if cur element matches first element 'T' ,or 'F'
                if (row[0] == row[c])
                    pattern += "T";
                else
                    pattern += "F";
            }
            mp[pattern]++;
        }
        int mxfreq = 0;
        for (auto &i : mp)
        {
            mxfreq = max(mxfreq, i.second);
        }
        return mxfreq;
    }
};
class P_1861 {
public:
    vector<vector<char>> rotateTheBox(vector<vector<char>> &box){
        /* Row by Row Brute force O(Mx(N^2))
        int m = box.size(),n = box[0].size();
        vector<vector<char>> rotated(n,vector<char>(m));
        //Transpose
        for(int i =0;i<n;i++){
            for(int j = 0;j < m;j++){
                rotated[i][j] = box[j][i];
            }
        }
        //Reverse each row of Transpose matrix
        for(int i = 0;i < n;i++){
            reverse(rotated[i].begin(),rotated[i].end());
        }
        for(int i = 0;i < m;i++){
            for(int j = n - 1;j >= 0;j--){
                if(rotated[j][i] == '.'){
                    int next_row_stone = -1;
                    for(int k = j - 1;k >= 0;k--){
                        if(rotated[k][i] == '*') break;
                        if(rotated[k][i] == '#'){next_row_stone = k;break;}
                    }
                    if(next_row_stone != -1){
                        rotated[next_row_stone][i] = '.';
                        rotated[j][i] = '#';
                    }
                }
            }
        }
    return rotated;*/
        // Row by Row Optimised O(M x N)
        int m = box.size(), n = box[0].size();
        vector<vector<char>> rotated(n, vector<char>(m));
        // Transpose
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                rotated[i][j] = box[j][i];
            }
        }
        // Reverse each row of Transpose matrix
        for (int i = 0; i < n; i++){
            reverse(rotated[i].begin(), rotated[i].end());
        }
        for (int i = 0; i < m; i++){
            int lowest = n - 1; // Lowest Row with Empty Cell
            for (int j = n - 1; j >= 0; j--){
                if (rotated[j][i] == '#'){
                   rotated[j][i] = '.';
                   rotated[lowest][i] = '#';
                   lowest--;
                }
                if(rotated[j][i] == '*')
                    lowest = j - 1;
            }
        }
        return rotated;
    }
};
class P_1975 {
public:
    long long maxMatrixSum(vector<vector<int>> &matrix) {
        int n = matrix.size(), minAbs = INT_MAX, neg_count = 0;
        long long sum = 0;
        for(int i = 0;i < n;i++){
            for(int j = 0;j < n;j++){
                if(matrix[i][j] < 0) {neg_count++;}
                minAbs = min(minAbs,abs(matrix[i][j]));
                sum += abs(matrix[i][j]);
            }
        }
        if(neg_count % 2 > 0){
            sum -= 2 * minAbs;
        }
        return sum;
    }
};
class P_773  {
private:
    vector<vector<int>> possible_dir = {{1, 3}, {0, 2, 4}, {1, 5}, 
                                        {0, 4}, {3, 5, 1}, {4, 2}};
    void dfs(string state,unordered_map<string,int> &vis,int zero_pos,int moves) {
        if(vis.count(state) && vis[state] <= moves){ return ;} // if already visited
        vis[state] = moves;
        //Try moving zero to each possible adjacent position
        for(int i:possible_dir[zero_pos]){
            swap(state[zero_pos],state[i]);
            dfs(state,vis,i,moves+1);
            swap(state[zero_pos],state[i]);
        }
    }
public:
    int slidingPuzzle_bfs(vector<vector<int>> & board){
        string final_state = "123450",start_state;
        for(int i = 0;i < 2;i++){
            for(int j = 0;j < 3;j++){
                start_state += to_string(board[i][j]);
            }
        }
        unordered_set<string> vis;
        queue<string> q;
        q.push(start_state);
        vis.insert(start_state);
        int moves = 0;

        while(!q.empty()){
            int s = q.size();
            while(s--){
                string cur_state = q.front();
                q.pop();

                if(cur_state == final_state) return moves;

                int zero_pos = cur_state.find('0');
                for(int newpos:possible_dir[zero_pos]){
                    string nextstate = cur_state;
                    swap(nextstate[zero_pos],nextstate[newpos]);

                    if(vis.count(nextstate)) continue;

                    vis.insert(nextstate);
                    q.push(nextstate);
                }
            }
            moves++;
        }
        return -1;
    }

    int slidingPuzzle_dfs(vector<vector<int>> &board){
        string final_state = "123450", cur_state;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cur_state += to_string(board[i][j]);
            }
        }
        unordered_map<string,int> vis;
        dfs(cur_state,vis,cur_state.find('0'),0);
        return vis.count(final_state)?vis[final_state]:-1;
    }
};
class P_2924 {
public:
    int findChampion(int n, vector<vector<int>> &edges){
        vector<int> indegree(n,0);
        for(auto i:edges) indegree[i[1]]++;
        int champ = -1,cnt = 0;
        for(int i = 0;i < n;i++){
            if(indegree[i] == 0){
                champ = i;
                cnt++;
            }
        }
        return (cnt > 1)?-1:champ;
    }
};
class P_3243 {
    public:
        int bfs(int n,vector<vector<int>>& adj){
            vector<bool> vis(n,false);
            queue<int> q;
            q.push(0);
            vis[0] = true;

            int cur_cnt = 1,nxt_cnt = 0,explored = 0;
            while(!q.empty()){
                for(int i = 0;i < cur_cnt;i++){
                    int cur = q.front();
                    q.pop();

                    if(cur == n-1) return explored;

                    for(auto &neighbor : adj[cur]){
                        if(vis[neighbor]) continue;

                        q.push(neighbor);
                        nxt_cnt++;
                        vis[neighbor] = true;
                    }
                }
                cur_cnt = nxt_cnt;
                nxt_cnt = 0;
                explored++;
            }
            return -1;
        }
        vector<int> shortestDistanceAfterQueries(int n, vector<vector<int>> &queries){
            vector<int> ans;
            vector<vector<int>> adjlist(n,vector<int>(0));
            for(int i = 0;i <= n-1;i++){ //initialise 0 -> n-1 Graph
                adjlist[i].push_back(i+1);
            }
            for(auto &i:queries){
                int u = i[0],v = i[1];
                adjlist[u].push_back(v);
                ans.push_back(bfs(n,adjlist));
            }
           // print_matrix(adjlist);
            return ans;
        }
};
class P_2290 {
public:
    /*You are given a 0 - indexed 2D integer array grid of size m x n.Each cell has one of two values :
    0 represents an empty cell,
    1 represents an obstacle that may be removed.
    You can move up,down, left, or right from and to an empty cell.Return the minimum number of obstacles to remove so you can move from the upper left corner(0, 0)
    to the lower right corner(m - 1, n - 1)
    Input: grid = [[0,1,1],[1,1,0],[1,1,0]]
    Output: 2
    Explanation: We can remove the obstacles at (0, 1) and (0, 2) to create a path from (0, 0) to (2, 2).
    It can be shown that we need to remove at least 2 obstacles, so we return 2.
    Note that there may be other ways to remove 2 obstacles to create a path.*/
    vector<vector<int>> dir = {{0,1},{0,-1},{1,0},{-1,0}}; //RLDU
    bool isvalid(vector<vector<int>>&grid,int r,int c){
        return r >= 0 && c >= 0 && r < grid.size() && c < grid[0].size();
    }
    int minimumObstacles(vector<vector<int>> &grid){
        int m = grid.size(),n = grid[0].size();
        vector<vector<int>> minobs(m,vector<int>(n,INT_MAX));
        minobs[0][0] = grid[0][0];
        priority_queue<vector<int>,vector<vector<int>>,greater<vector<int>>> pq;
        pq.push({minobs[0][0],0,0}); //obstacles count,row,col
        while(!pq.empty()){
            vector<int> cur = pq.top();
            pq.pop();
            int obstacles = cur[0],r = cur[1],c = cur[2];
            if(r == m-1 && c == n-1) return obstacles;

            for(vector<int> &d:dir){
                int new_r = r + d[0],new_c = c + d[1];
                if(isvalid(grid,new_r,new_c)){
                    int newobs = obstacles + grid[new_r][new_c];

                    if(newobs < minobs[new_r][new_c]){
                        minobs[new_r][new_c] = newobs;
                        pq.push({newobs,new_r,new_c});
                    }
                }
            }
        }
        return minobs[m-1][n-1];
    }
};
class P_2577 {
    /*You are given m x n matrix grid consisting of non-negative integers where grid[row][col] represents the minimum time required to be able to visit the cell (row, col),
    which means you can visit the cell (row, col) only when the time you visit it is greater than or equal to grid[row][col].
    You are standing in the top-left cell of the matrix in the 0th second, and you must move to any adjacent cell in the four directions: up, down, left, and right.
    Each move you make takes 1 second.
    Return the minimum time required in which you can visit the bottom-right cell of the matrix. If you cannot visit the bottom-right cell, then return -1.
    Input: grid = [[0,1,3,2],[5,1,2,5],[4,3,8,6]]
    Output: 7
    Explanation: One of the paths that we can take is the following:
    - at t = 0, we are on the cell (0,0).
    - at t = 1, we move to the cell (0,1). It is possible because grid[0][1] <= 1.
    - at t = 2, we move to the cell (1,1). It is possible because grid[1][1] <= 2.
    - at t = 3, we move to the cell (1,2). It is possible because grid[1][2] <= 3.
    - at t = 4, we move to the cell (1,1). It is possible because grid[1][1] <= 4.
    - at t = 5, we move to the cell (1,2). It is possible because grid[1][2] <= 5.
    - at t = 6, we move to the cell (1,3). It is possible because grid[1][3] <= 6.
    - at t = 7, we move to the cell (2,3). It is possible because grid[2][3] <= 7.
    The final time is 7. It can be shown that it is the minimum time possible.*/
public:
    bool isvalid(vector<vector<bool>> &vis,int r,int c){
        return r >= 0 && c >= 0 && r < vis.size() && c < vis[0].size() && !vis[r][c];
    }
    int minimumTime(vector<vector<int>> &grid){
        if(grid[0][1] > 1 && grid[1][0] > 1) return -1; //if initial moverequire > 1 then impossible to reach end
        int r = grid.size(),c = grid[0].size();
        vector<vector<int>> dir = {{0,1},{0,-1},{1,0},{-1,0}};
        vector<vector<bool>> vis(r,vector<bool>(c,false));
        priority_queue<vector<int>,vector<vector<int>>,greater<>>pq; // stores time,r,c
        pq.push({grid[0][0],0,0});
        
        while(!pq.empty()){
            auto cur = pq.top();
            pq.pop();
            int time = cur[0],row = cur[1],col = cur[2];

            if(row == r - 1 && col == c - 1) return time; // target reached

            if(vis[row][col]) continue; //skip if visited
            vis[row][col] = true;
            //try all 4 directions
            for(auto &i:dir){
                int nxt_r = row + i[0],nxt_c = col + i[1];
                if(!isvalid(vis,nxt_r,nxt_c)){ continue; }

                int wait_time = ((grid[nxt_r][nxt_c] - time) %2 == 0)?1:0;
                int nxt_time = max(grid[nxt_r][nxt_c] + wait_time,time + 1);
                pq.push({nxt_time,nxt_r,nxt_c});
            }
        }
        return -1;
    }
};
class P_2097 {
    /*You are given a 0-indexed 2D integer array pairs where pairs[i] = [starti, endi]. An arrangement of pairs is valid if for every index i where 1 <= i < pairs.length,
     we have endi-1 == starti.Return any valid arrangement of pairs. Note: The inputs will be generated such that there exists a valid arrangement of pairs.
     Example 1:
     Input: pairs = [[5,1],[4,5],[11,9],[9,4]]
     Output: [[11,9],[9,4],[4,5],[5,1]]
     Explanation:
     This is a valid arrangement since endi-1 always equals starti.
     end0 = 9 == 9 = start1
     end1 = 4 == 4 = start2
     end2 = 5 == 5 = start3
     
     Example 2:
     Input: pairs = [[1,3],[3,2],[2,1]]
     Output: [[1,3],[3,2],[2,1]]
     Explanation:
     This is a valid arrangement since endi-1 always equals starti.
     end0 = 3 == 3 = start1
     end1 = 2 == 2 = start2
     The arrangements [[2,1],[1,3],[3,2]] and [[3,2],[2,1],[1,3]] are also valid.
     
     Example 3:
     Input: pairs = [[1,2],[1,3],[2,1]]
     Output: [[1,2],[2,1],[1,3]]
     Explanation:
     This is a valid arrangement since endi-1 always equals starti.
     end0 = 2 == 2 = start1
     end1 = 1 == 1 = start2 */
public:
    vector<vector<int>> validArrangement(vector<vector<int>> &pairs){
        unordered_map<int, deque<int>> adjacencyMatrix;
        unordered_map<int, int> inDegree, outDegree;

        // Build the adjacency list and track in-degrees and out-degrees
        for (const auto &pair : pairs)
        {
            int start = pair[0], end = pair[1];
            adjacencyMatrix[start].push_back(end);
            outDegree[start]++;
            inDegree[end]++;
        }

        vector<int> result;

        // Helper lambda function for DFS traversal,
        // you can make a seperate private function also
        function<void(int)> visit = [&](int node)
        {
            while (!adjacencyMatrix[node].empty())
            {
                int nextNode = adjacencyMatrix[node].front();
                adjacencyMatrix[node].pop_front();
                visit(nextNode);
            }
            result.push_back(node);
        };

        // Find the start node (outDegree == 1 + inDegree )
        int startNode = -1;
        for (const auto &entry : outDegree)
        {
            int node = entry.first;
            if (outDegree[node] == inDegree[node] + 1)
            {
                startNode = node;
                break;
            }
        }

        // If no such node exists, start from the first pair's first element
        if (startNode == -1)
        {
            startNode = pairs[0][0];
        }

        // Start DFS traversal
        visit(startNode);

        // Reverse the result since DFS gives us the path in reverse
        reverse(result.begin(), result.end());

        // Construct the result pairs
        vector<vector<int>> pairedResult;
        for (int i = 1; i < result.size(); ++i)
        {
            pairedResult.push_back({result[i - 1], result[i]});
        }

        return pairedResult;
    }
};
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    P_2097 s;
    //vector<vector<int>> ar = {{1,2,3},{4,0,5}};

    vector<vector<int>> pairs = {{5,1},{4,5},{9,4},{11,9}};
    vector<vector<int>> ans = s.validArrangement(pairs);
    print_matrix(ans);
    
    return 0;
}