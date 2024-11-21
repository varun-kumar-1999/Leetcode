#include<bits/stdc++.h>
using namespace std;
class sol{
    public:
    vector<vector<int>> threesum_optimised1(vector<int>& ar){
        int n = ar.size();
        set<vector<int>> st;
        vector<int> tmp(3);
        for(int i = 0;i<n;i++){
            set<int> hashset;
            for(int j = i+1;j<n;j++){
                int third = -(ar[i]+ar[j]);
                if(hashset.find(third) != hashset.end()){
                    tmp = {ar[i],ar[j],third};
                    sort(tmp.begin(),tmp.end());
                    st.insert(tmp);
                    tmp.clear();
                }
                hashset.insert(ar[j]);
            }
            
        }
    vector<vector<int>> ans(st.begin(),st.end()); 
    return ans;
    }
    vector<vector<int>> threesum_optimised2(vector<int>& ar){  // -2 -2 -2 0 0 0 1 1 1
       vector<vector<int>> ans; 
        int n = ar.size();
        sort(ar.begin(),ar.end());
        for(int start = 0;start < n;start++){
            if(start > 0 && ar[start] == ar[start-1]) continue;
            int sec = start+1;
            int end = n-1;
            while(sec < end){
                int sum = ar[start] + ar[sec] + ar[end];
                if(sum < 0){
                    sec++;
                }
                else if(sum > 0){
                    end--;
                }
                else{
                    vector<int> tmp = {ar[start],ar[sec],ar[end]};
                    ans.push_back(tmp);
                    sec++,end--;
                    while(sec < end && ar[sec] == ar[sec-1]) sec++;
                    while(sec < end && ar[end] == ar[end+1]) end--;
                }
            }
        }
    return ans;
    
    }
    vector<vector<int>> threesum_brute(vector<int>& ar){ 
        vector<vector<int>> ans;
        set<vector<int>> s;
        int n = ar.size();
        vector<int> tmp;
        for(int i = 0;i<n;i++){
            for(int j = i+1;j<n;j++){
                for(int k = j+1;k<n;k++){
                    if(ar[i] + ar[j] + ar[k] == 0){
                        tmp = {ar[i],ar[j],ar[k]};
                        sort(tmp.begin(),tmp.end());
                        s.insert(tmp);
                        tmp.clear();
                    }
                }
            }
        }
        for(auto i:s)
            ans.push_back(i);
        return ans;
    }
};
int main(){
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
vector<int> a  = {-1,0,1,2,-1,-1};
sol s;
vector<vector<int>> ans = s.threesum_optimised2(a);
for(int i = 0;i<ans.size();i++){
    for(int j:ans[i]){
        cout<<j<<" ";
    }
    cout<<"\n";
}
return 0;
}